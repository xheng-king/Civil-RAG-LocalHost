# retriever_generator.py
import os
import chromadb
from openai import OpenAI
from typing import List, Dict, Any
from database_manager import DatabaseManager
import time
import json
from datetime import datetime
import csv
import glob
import math
from collections import Counter
from tqdm import tqdm

from settings import base_url_set, embedding_model, embedding_API_key, rerank_model, rerank_base_url, rerank_API_key, llm, llm_base_url, llm_API_key

class QwenRetrieverGenerator:
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
        
        # 重排序 API key (用于 requests 手动调用)
        self.rerank_api_key = rerank_API_key
        
        self.chroma_client = chromadb.PersistentClient(path="../data/vectorstore")
        self.db_manager = DatabaseManager()
        self.initial_retrieve_k = 5
        self.final_top_k = 3
        self.log_file_path = "../query_log.md"

    def _log_interaction(self, user_input: str, response: str):
        markdown_content = f"Q：{user_input}\nA：{response}\n\n"
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
    
    def retrieve_documents(self, query_text: str) -> List[Dict[str, Any]]:
        query_embedding = [self.embed_query(query_text)]
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.initial_retrieve_k,
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
                'score': 1.0 - dist,          # 新增：余弦相似度分数，供评估使用
                'rerank_score': None
            })
        
        return retrieved_docs
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        print(f"正在使用重排序API对{len(documents)}个候选片段进行重排序...")

        try:
            import requests
            
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
                        updated_doc['rerank_rank'] = len(reranked_docs) + 1 
                        reranked_docs.append(updated_doc)
                    else:
                        print(f"警告: 重排序API返回了无效的索引 {original_index}")
                
                print("重排序成功完成")
                return reranked_docs
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

        print("重排序失败，使用初始检索顺序")
        documents.sort(key=lambda x: 1.0 / (x['initial_distance'] + 0.0001), reverse=True) 
        return documents
    
    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        context_str = "\n\n".join([
            f"参考信息 #{doc['rerank_rank']} (相关性分数: {doc['rerank_score']:.4f}, 来源: {doc['metadata'].get('source', '未知')}):\n{doc['content']}" 
            for i, doc in enumerate(contexts)
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
                print(f"无法从模型响应中提取数字: {response}")
                return 10
        except Exception as e:
            print(f"调用模型判断相关性时出错: {e}")
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
                print(f"无法从模型响应中提取等级: {response}")
                return 0
        except Exception as e:
            print(f"调用模型评估相关性时出错: {e}")
            return 0
            
    def query(self, user_input: str) -> str:
        print(f"用户输入: {user_input}")
        
        print(f"正在检索前 {self.initial_retrieve_k} 个候选文档...")
        candidate_docs = self.retrieve_documents(user_input)
        
        if not candidate_docs:
            response = "抱歉，没有找到相关文档。"
            self._log_interaction(user_input, response)
            return response
        
        print("\n初始检索结果（按向量相似度）:")
        for i, doc in enumerate(candidate_docs):
            print(f"  #{i+1} 距离: {doc['initial_distance']:.4f} | 来源: {doc['metadata'].get('source', '未知')}")
        
        reranked_docs = self.rerank_documents(user_input, candidate_docs)
        
        print("\n重排序后结果（按相关性分数）:")
        for i, doc in enumerate(reranked_docs):
            print(f"  #{doc['rerank_rank']} 分数: {doc['rerank_score']:.4f} | 来源: {doc['metadata'].get('source', '未知')}")
        
        final_docs = reranked_docs[:self.final_top_k]
        print(f"\n选择前 {self.final_top_k} 个最相关片段用于生成回答")
        
        print("正在基于精选片段生成答案...")
        answer = self.generate_answer(user_input, final_docs)
        
        self._log_interaction(user_input, answer)
        
        return answer