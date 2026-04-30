# retriever_generator.py
import os
import chromadb
from openai import OpenAI
from typing import List, Dict, Any, Callable, Optional, Tuple
from database_manager import DatabaseManager
import time
import json
from datetime import datetime
import csv
import glob
import math
from collections import Counter
from tqdm import tqdm
import requests

# 导入设置
from settings import (
    base_url_set, embedding_model, embedding_API_key, 
    rerank_model, rerank_base_url, rerank_API_key, 
    llm, llm_base_url, llm_API_key,
    # 新增导入
    ENABLE_ADAPTIVE_RETRIEVAL, MAX_RETRIEVAL_ROUNDS, 
    RETRIEVAL_STEP_SIZE, RERANK_OUTPUT_STEP_SIZE,
    BASE_INITIAL_RETRIEVE_K, BASE_FINAL_TOP_K
)

class RetrieverGenerator:
    def __init__(self):
        # Embedding client
        if not embedding_API_key:
            raise ValueError("settings.py 中的 embedding_API_key 未设置")
        self.embedding_client = OpenAI(
            api_key=embedding_API_key,
            base_url=base_url_set
        )
        
        # LLM client
        if not llm_API_key:
            raise ValueError("settings.py 中的 llm_API_key 未设置")
        self.llm_client = OpenAI(
            api_key=llm_API_key,
            base_url=llm_base_url
        )
        
        # 重排序 API key
        self.rerank_api_key = rerank_API_key
        
        self.chroma_client = chromadb.PersistentClient(path="../data/vectorstore")
        self.db_manager = DatabaseManager()
        
        # 使用 settings 中的基础配置初始化默认值
        self.initial_retrieve_k = BASE_INITIAL_RETRIEVE_K
        self.final_top_k = BASE_FINAL_TOP_K
        
        self.log_file_path = "../query_log.md"
        
        # 当前运行时状态
        self.collection = None

    def _log_interaction(self, user_input: str, response: str, round_num: int = 1, status: str = "Final"):
        markdown_content = f"--- Round {round_num} ({status}) ---\nQ：{user_input}\nA：{response}\n\n"
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(markdown_content)
        except Exception as e:
            print(f"记录日志时出错: {e}")

    def select_collection(self):
        collection_names = self.db_manager.list_collections()
        
        if not collection_names:
            print("没有可用的集合，请先创建或索引一些数据")
            return None
        
        print("\n请选择要查询的集合:")
        for i, name in enumerate(collection_names, 1):
            print(f"  {i}. {name}")
        
        while True:
            try:
                choice = int(input(f"\n请选择 (1-{len(collection_names)}): "))
                if 1 <= choice <= len(collection_names):
                    selected_collection_name = collection_names[choice - 1]
                    
                    try:
                        self.collection = self.chroma_client.get_collection(name=selected_collection_name)
                        print(f"已选择集合: {selected_collection_name}")
                        return selected_collection_name
                    except Exception as e:
                        print(f"获取集合时出错: {e}")
                        return None
                else:
                    print(f"请输入 1 到 {len(collection_names)} 之间的数字")
            except ValueError:
                print("请输入有效的数字")
            except EOFError:
                print("\n操作取消")
                return None
    
    def embed_query(self, query_text: str) -> List[float]:
        response = self.embedding_client.embeddings.create(
            model=embedding_model,
            input=query_text
        )
        return response.data[0].embedding
    
    def retrieve_documents(self, query_text: str, k: int = None) -> List[Dict[str, Any]]:
        """
        检索文档，支持动态指定 k 值。如果 k 为 None，则使用实例默认的 initial_retrieve_k
        """
        if k is None:
            k = self.initial_retrieve_k
            
        query_embedding = [self.embed_query(query_text)]
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        retrieved_docs = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            retrieved_docs.append({
                'id': i,
                'content': doc,
                'metadata': meta,
                'initial_distance': dist,
                'score': 1.0 - dist,          # 余弦相似度分数
                'rerank_score': None
            })
        
        return retrieved_docs
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]], top_n: int = None) -> List[Dict[str, Any]]:
        """
        重排序文档，支持动态指定最终保留的 top_n 数量
        """
        if top_n is None:
            top_n = self.final_top_k

        if not documents:
            return []

        # 如果请求的 top_n 大于文档总数，则调整为文档总数
        actual_top_n = min(top_n, len(documents))
        if actual_top_n <= 0:
            return []

        # 静默模式可选，这里为了调试保留打印，正式评估可注释
        # print(f"正在使用重排序API对{len(documents)}个候选片段进行重排序，保留前{actual_top_n}个...")

        try:
            headers = {
                "Authorization": f"Bearer {self.rerank_api_key}",
                "Content-Type": "application/json"
            }

            texts_to_rerank = [doc['content'] for doc in documents]
            payload = {
                "model": rerank_model,
                "documents": texts_to_rerank,
                "query": query,
                "top_n": len(documents), # API通常返回所有排序结果，我们在本地截取
            }
            
            response = requests.post(
                rerank_base_url,
                headers=headers,
                json=payload
            )

            response.raise_for_status() 
            result = response.json()

            if 'results' in result:
                api_results = result['results']
                reranked_docs = []
                
                # api_results 通常已经按相关性降序排列
                for rank_data in api_results:
                    original_index = rank_data['index']
                    relevance_score = rank_data['relevance_score']
                    
                    if original_index < len(documents):
                        original_doc = documents[original_index]
                        updated_doc = original_doc.copy()
                        updated_doc['rerank_score'] = relevance_score
                        reranked_docs.append(updated_doc)
                
                # 截取前 top_n 个
                final_reranked = reranked_docs[:actual_top_n]
                
                # 重新分配显示用的 rank (1-based)
                for idx, doc in enumerate(final_reranked):
                    doc['rerank_rank'] = idx + 1

                # print("重排序成功完成")
                return final_reranked
            else:
                print(f"API响应格式不符合预期: {result}")
                
        except requests.exceptions.HTTPError as e:
            print(f"HTTP错误: {e.response.status_code}, {e.response.text}")
        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
        except KeyError as e:
            print(f"解析响应时缺少键: {e}")
        except Exception as e:
            print(f"重排序过程中发生未知异常: {e}")

        # 失败降级处理：按初始距离排序并截取
        print("重排序失败，使用初始检索顺序")
        documents.sort(key=lambda x: 1.0 / (x['initial_distance'] + 0.0001), reverse=True) 
        return documents[:actual_top_n]
    
    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        if not contexts:
            return "抱歉，没有找到相关文档。"

        context_str = "\n\n".join([
            f"参考信息 #{doc['rerank_rank']} (相关性分数: {doc['rerank_score']:.4f}, 来源: {doc['metadata'].get('source', '未知')}):\n{doc['content']}" 
            for doc in contexts
        ])
        
        prompt = f"""基于以下数据库内容，回答用户的问题。

数据库内容：
{context_str}

用户问题：
{query}

回答要求：
1.若数据库内容中存在用户问题的相关回答，则直接简明扼要地回答问题
2.若数据库中不存在用户问题的相关解答，提示用户查询结果没有相关内容
"""
        
        try:
            completion = self.llm_client.chat.completions.create(
                model=llm,
                messages=[
                    {"role": "system", "content": "你是一个专业的知识助手，能够基于提供的多段上下文信息回答用户的问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成答案时出错: {e}")
            # 简化Prompt重试
            simple_context = "\n\n".join([doc['content'] for doc in contexts])
            simple_prompt = f"基于以下信息回答问题:\n\n{simple_context}\n\n问题: {query}"
            
            try:
                completion = self.llm_client.chat.completions.create(
                    model=llm,
                    messages=[{"role": "user", "content": simple_prompt}],
                    max_tokens=600
                )
                return completion.choices[0].message.content.strip()
            except:
                return "抱歉，在生成答案时遇到问题。相关信息可能不足。"

    # --- 辅助评估函数 (保留原样，用于内部调试或作为默认评估器) ---
    def get_relevance_rank(self, query, documents):
        prompt = f"""请判断以下哪个文档包含了用户问题的答案。如果文档中包含答案，请返回该文档的序号（1-{len(documents)}）；如果没有文档包含答案，请返回 0。

用户问题：{query}

文档列表：
"""
        for i, doc in enumerate(documents, 1):
            prompt += f"\n文档 #{i}:\n{doc['content'][:1000]}"
        
        prompt += "\n\n请只返回一个数字，表示包含答案的文档序号，如果没有包含答案的文档，请返回 0。"

        try:
            completion = self.llm_client.chat.completions.create(
                model=llm,
                messages=[
                    {"role": "system", "content": "你是一个专业的相关性评估助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            response = completion.choices[0].message.content.strip()
            try:
                rank = int(''.join(filter(str.isdigit, response)))
                if rank == 0:
                    return 10
                return rank
            except:
                return 10
        except Exception as e:
            return 10

    def get_relevance_grade(self, query, document):
        prompt = f"""请评估以下文档与用户问题的相关性等级，等级范围为0-4：
0: 完全不相关
1: 轻微相关
2: 中等相关
3: 高度相关
4: 包含答案

用户问题：{query}

文档内容：
{document['content'][:1000]}

请只返回一个数字 (0-4) 表示相关性等级。"""

        try:
            completion = self.llm_client.chat.completions.create(
                model=llm,
                messages=[
                    {"role": "system", "content": "你是一个专业的相关性评估助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            response = completion.choices[0].message.content.strip()
            try:
                grade = int(''.join(filter(str.isdigit, response)))
                return min(max(0, grade), 4)
            except:
                return 0
        except Exception as e:
            return 0
            
    def _execute_single_round(self, user_input: str, initial_k: int, final_top_k: int) -> Tuple[str, List[Dict], List[Dict]]:
        """
        执行单轮检索-重排序-生成流程
        返回: (answer, final_docs_used_for_llm, candidate_docs_before_rerank)
        """
        # print(f"[Round] 初始召回: {initial_k}, 重排序后保留: {final_top_k}")
        
        # 1. Retrieve
        candidate_docs = self.retrieve_documents(user_input, k=initial_k)
        
        if not candidate_docs:
            return "抱歉，没有找到相关文档。", [], []
        
        # 2. Rerank
        # 注意：这里我们让 rerank 处理所有候选，但只返回前 final_top_k 个给 LLM
        # 但是为了计算 MRR/NDCG，我们需要知道所有候选文档的重排序分数
        # 所以这里我们调用 rerank_documents 获取所有排序后的文档（或者至少是带分数的）
        # 修改：让 rerank_documents 返回所有带分数的文档，然后在外部截取给 LLM 的部分
        # 但为了兼容现有接口，我们让 rerank_documents 返回截取后的，但我们需要所有文档的分数
        
        # 策略：先获取所有文档的重排序分数，再截取
        all_reranked = self._rerank_all_documents(user_input, candidate_docs)
        
        # 截取给 LLM 的部分
        final_docs_for_llm = all_reranked[:final_top_k]
        # 重新分配 rank 给 LLM 用的部分
        for idx, doc in enumerate(final_docs_for_llm):
            doc['rerank_rank'] = idx + 1
            
        # 3. Generate
        answer = self.generate_answer(user_input, final_docs_for_llm)
        
        return answer, final_docs_for_llm, candidate_docs

    def _rerank_all_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        内部辅助函数：对所有文档进行重排序并返回带分数的完整列表（不截取）
        """
        if not documents:
            return []

        try:
            headers = {
                "Authorization": f"Bearer {self.rerank_api_key}",
                "Content-Type": "application/json"
            }

            texts_to_rerank = [doc['content'] for doc in documents]
            payload = {
                "model": rerank_model,
                "documents": texts_to_rerank,
                "query": query,
                "top_n": len(documents), 
            }
            
            response = requests.post(
                rerank_base_url,
                headers=headers,
                json=payload
            )

            response.raise_for_status() 
            result = response.json()

            if 'results' in result:
                api_results = result['results']
                reranked_docs = []
                
                for rank_data in api_results:
                    original_index = rank_data['index']
                    relevance_score = rank_data['relevance_score']
                    
                    if original_index < len(documents):
                        original_doc = documents[original_index]
                        updated_doc = original_doc.copy()
                        updated_doc['rerank_score'] = relevance_score
                        reranked_docs.append(updated_doc)
                
                return reranked_docs
            else:
                print(f"API响应格式不符合预期: {result}")
                
        except Exception as e:
            print(f"重排序过程中发生未知异常: {e}")

        # 失败降级
        documents.sort(key=lambda x: 1.0 / (x['initial_distance'] + 0.0001), reverse=True) 
        return documents

    def query(self, user_input: str, evaluator_func: Optional[Callable[[str], bool]] = None) -> Tuple[str, List[Dict], List[Dict]]:
        """
        主查询入口。
        
        Args:
            user_input: 用户问题
            evaluator_func: 可选。评估函数。
            
        Returns:
            Tuple: (answer, final_docs_used_for_llm, candidate_docs_before_rerank)
        """
        # print(f"用户输入: {user_input}")
        
        # 如果没有提供评估函数，或者未启用自适应检索，则执行标准单次流程
        if not ENABLE_ADAPTIVE_RETRIEVAL or evaluator_func is None:
            answer, final_docs, candidates = self._execute_single_round(
                user_input, 
                self.initial_retrieve_k, 
                self.final_top_k
            )
            self._log_interaction(user_input, answer, round_num=1, status="Standard")
            return answer, final_docs, candidates

        # --- 自适应重试逻辑 ---
        current_initial_k = self.initial_retrieve_k
        current_final_top_k = self.final_top_k
        last_answer = ""
        last_final_docs = []
        last_candidates = []
        
        for round_num in range(1, MAX_RETRIEVAL_ROUNDS + 1):
            # print(f"\n=== 开始第 {round_num} 轮查询 ===")
            
            # 执行单轮
            answer, final_docs, candidates = self._execute_single_round(user_input, current_initial_k, current_final_top_k)
            last_answer = answer
            last_final_docs = final_docs
            last_candidates = candidates
            
            # 调用外部评估函数
            is_correct = evaluator_func(answer)
            
            if is_correct:
                # print(f"✅ 第 {round_num} 轮评估通过。")
                self._log_interaction(user_input, answer, round_num=round_num, status="Success")
                return answer, final_docs, candidates
            else:
                # print(f"❌ 第 {round_num} 轮评估失败 (INCORRECT)。")
                self._log_interaction(user_input, answer, round_num=round_num, status="Failed/Retry")
                
                # 如果不是最后一轮，准备下一轮的参数
                if round_num < MAX_RETRIEVAL_ROUNDS:
                    current_initial_k += RETRIEVAL_STEP_SIZE
                    current_final_top_k += RERANK_OUTPUT_STEP_SIZE
                    # print(f"-> 调整参数: 下一轮初始召回={current_initial_k}, 重排序保留={current_final_top_k}")
                else:
                    # print(f"⚠️ 已达到最大重试轮次 ({MAX_RETRIEVAL_ROUNDS})，返回最后一轮结果。")
                    pass

        # 循环结束，返回最后一轮的结果
        self._log_interaction(user_input, last_answer, round_num=MAX_RETRIEVAL_ROUNDS, status="MaxRetriesReached")
        return last_answer, last_final_docs, last_candidates