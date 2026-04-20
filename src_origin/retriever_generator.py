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


class QwenRetrieverGenerator:
    def __init__(self):
        # 设置 Qwen API
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 初始化向量数据库客户端
        self.chroma_client = chromadb.PersistentClient(path="../data/vectorstore")
        self.db_manager = DatabaseManager()
        self.initial_retrieve_k = 5  # 初始召回5个候选片段
        self.final_top_k = 3  # 重排序后使用3个最相关片段
        # 设置日志文件路径为项目根目录
        self.log_file_path = "../query_log.md"

    def _log_interaction(self, user_input: str, response: str):
        """记录用户输入和模型响应到日志文件"""
        markdown_content = f"Q：{user_input}\nA：{response}\n\n"

        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(markdown_content)
        except Exception as e:
            print(f"记录日志时出错: {e}")

    def select_collection(self):
        """让用户选择集合"""
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
                    
                    # 检查集合是否存在
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
        """使用 Qwen 嵌入模型将查询文本转换为向量"""
        response = self.client.embeddings.create(
            model="text-embedding-v2",
            input=query_text
        )
        return response.data[0].embedding
    
    def retrieve_documents(self, query_text: str) -> List[Dict[str, Any]]:
        """召回k个文档(ChromaDB 默认为余弦相似度)"""
        # 将查询转换为嵌入向量
        query_embedding = [self.embed_query(query_text)]
        
        # 执行相似度搜索，召回候选（5个）
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.initial_retrieve_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # 整理结果
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
                'rerank_score': None
            })
        
        return retrieved_docs
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用rerank重排序模型对候选文档进行重排序
        """
        if not documents:
            return []

        print(f"正在使用DashScope重排序API对{len(documents)}个候选片段进行重排序...")

        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.client.api_key}", # 使用openai client中的api_key
                "Content-Type": "application/json"
            }

            # 准备请求体
            texts_to_rerank = [doc['content'] for doc in documents]
            payload = {
                "model": "qwen3-rerank",
                "documents": texts_to_rerank,
                "query": query,
                "top_n": len(documents), # 请求返回所有文档的排序结果
                # "instruct": "Given a web search query, retrieve relevant passages that answer the query." # 可选
            }
            
            # 发送到正确的API端点
            response = requests.post(
                "https://dashscope.aliyuncs.com/compatible-api/v1/reranks", # 注意这里是 compatible-api/v1/reranks
                headers=headers,
                json=payload
            )

            # 检查HTTP状态码
            response.raise_for_status() 

            # 解析JSON响应
            result = response.json()

            # 检查响应中是否有预期的字段
            # 根据您的日志，实际格式是 {"results": [...], "object": "..."}，而不是 {"output": {"results": [...]}}
            if 'results' in result: # 直接检查 'results' 而不是 'output']['results'
                api_results = result['results'] # 获取结果列表
                
                # 创建一个新列表来存放重排序后的文档
                reranked_docs = []
                
                # api_results 已经是按相关性分数降序排列的
                for rank_data in api_results:
                    original_index = rank_data['index'] # 获取原始文档的索引
                    relevance_score = rank_data['relevance_score'] # 获取相关性分数
                    
                    # 获取原始文档对象
                    if original_index < len(documents): # 确保索引有效
                        original_doc = documents[original_index]
                        
                        # 创建一个新的字典，包含原始信息和新的重排序分数
                        updated_doc = original_doc.copy()
                        updated_doc['rerank_score'] = relevance_score
                        # 可以添加排名信息（因为api_results是排序后的，所以索引就是排名）
                        updated_doc['rerank_rank'] = len(reranked_docs) + 1 
                        
                        reranked_docs.append(updated_doc)
                    else:
                        print(f"警告: 重排序API返回了无效的索引 {original_index}")
                
                print("重排序成功完成")
                return reranked_docs
            else:
                print(f"API响应格式不符合预期: {result}")
                # 如果响应格式不对，也回退
                
        except requests.exceptions.HTTPError as e:
            print(f"HTTP错误: {e.response.status_code}, {e.response.text}")
        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
        except KeyError as e:
            print(f"解析响应时缺少键: {e}")
        except Exception as e:
            print(f"重排序过程中发生未知异常: {e}")

        # 如果重排序失败，回退到初始检索顺序
        print("重排序失败，使用初始检索顺序")
        # 按原始距离的倒数排序（即距离越小，分数越高）
        documents.sort(key=lambda x: 1.0 / (x['initial_distance'] + 0.0001), reverse=True) 
        return documents
    
    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """使用 Qwen 大模型生成答案（支持多个上下文）"""
        # 将多个上下文整合为一个字符串，添加序号和分隔符
        # 包含重排序分数信息
        context_str = "\n\n".join([
            f"参考信息 #{doc['rerank_rank']} (相关性分数: {doc['rerank_score']:.4f}, 来源: {doc['metadata'].get('source', '未知')}):\n{doc['content']}" 
            for i, doc in enumerate(contexts)
        ])
        
        # 设计提示词模板 - 强调使用所有相关信息
        prompt = f"""基于以下数据库内容，回答用户的问题。

数据库内容：
{context_str}

用户问题：
{query}

回答要求：
1.若数据库内容中存在用户问题的相关回答，则直接简明扼要地回答问题
2.若数据库中不存在用户问题的相关解答，提示用户查询结果没有相关内容
"""
        
        # 调用 Qwen 大模型
        try:
            completion = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的知识助手，能够基于提供的多段上下文信息回答用户的问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 降低温度以提高回答一致性
                max_tokens=800
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成答案时出错: {e}")
            # 出错时尝试简化请求
            simple_context = "\n\n".join([doc['content'] for doc in contexts])
            simple_prompt = f"基于以下信息回答问题:\n\n{simple_context}\n\n问题: {query}"
            
            try:
                completion = self.client.chat.completions.create(
                    model="qwen-turbo",
                    messages=[{"role": "user", "content": simple_prompt}],
                    max_tokens=600
                )
                return completion.choices[0].message.content.strip()
            except:
                return "抱歉，在生成答案时遇到问题。相关信息可能不足。"
    
    def get_relevance_rank(self, query, documents):
        """
        使用 qwen-turbo 判断第一个相关文档的排名
        返回第一个相关文档的排名，如果没有找到相关文档，返回 10
        """
        # 创建一个提示词，让模型判断每个文档是否包含查询的答案
        prompt = f"""请判断以下哪个文档包含了用户问题的答案。如果文档中包含答案，请返回该文档的序号（1-{len(documents)}）；如果没有文档包含答案，请返回 0。

用户问题：{query}

文档列表：
"""
        
        for i, doc in enumerate(documents, 1):
            prompt += f"\n文档 #{i}:\n{doc['content'][:1000]}"  # 只取前1000个字符，避免超出token限制
        
        prompt += "\n\n请只返回一个数字，表示包含答案的文档序号，如果没有包含答案的文档，请返回 0。"

        try:
            completion = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的相关性评估助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            response = completion.choices[0].message.content.strip()
            # 尝试提取数字
            try:
                rank = int(''.join(filter(str.isdigit, response)))
                if rank == 0:  # 没有找到相关文档
                    return 10
                return rank
            except:
                print(f"无法从模型响应中提取数字: {response}")
                return 10  # 默认认为没有找到相关文档
        except Exception as e:
            print(f"调用模型判断相关性时出错: {e}")
            return 10  # 出错时默认认为没有找到相关文档

    def get_relevance_grade(self, query, document):
        """
        使用 qwen-turbo 评估文档与查询的相关性等级 (0-4)
        0: 完全不相关
        1: 轻微相关
        2: 中等相关
        3: 高度相关
        4: 包含答案
        """
        prompt = f"""请评估以下文档与用户问题的相关性等级，等级范围为0-4：
0: 完全不相关
1: 轻微相关
2: 中等相关
3: 高度相关
4: 包含答案

用户问题：{query}

文档内容：
{document['content'][:1000]}  # 只取前1000个字符

请只返回一个数字 (0-4) 表示相关性等级。"""

        try:
            completion = self.client.chat.completions.create(
                model="qwen-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的相关性评估助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            response = completion.choices[0].message.content.strip()
            # 尝试提取数字
            try:
                grade = int(''.join(filter(str.isdigit, response)))
                return min(max(0, grade), 4)  # 确保在0-4范围内
            except:
                print(f"无法从模型响应中提取等级: {response}")
                return 0  # 默认完全不相关
        except Exception as e:
            print(f"调用模型评估相关性时出错: {e}")
            return 0  # 出错时默认完全不相关
    def query(self, user_input: str) -> str:
        """完整的 RAG 查询流程（包含重排序）"""
        print(f"用户输入: {user_input}")
        
        # 1. 检索候选文档
        print(f"正在检索前 {self.initial_retrieve_k} 个候选文档...")
        candidate_docs = self.retrieve_documents(user_input)
        
        if not candidate_docs:
            response = "抱歉，没有找到相关文档。"
            self._log_interaction(user_input, response)
            return response
        
        # 2. 显示初始检索结果
        print("\n初始检索结果（按向量相似度）:")
        for i, doc in enumerate(candidate_docs):
            print(f"  #{i+1} 距离: {doc['initial_distance']:.4f} | 来源: {doc['metadata'].get('source', '未知')}")
        
        # 3. 对候选文档进行重排序
        reranked_docs = self.rerank_documents(user_input, candidate_docs)
        
        # 4. 显示重排序结果
        print("\n重排序后结果（按相关性分数）:")
        for i, doc in enumerate(reranked_docs):
            print(f"  #{doc['rerank_rank']} 分数: {doc['rerank_score']:.4f} | 来源: {doc['metadata'].get('source', '未知')}")
        
        # 5. 选择最终使用的top-k文档
        final_docs = reranked_docs[:self.final_top_k]
        print(f"\n选择前 {self.final_top_k} 个最相关片段用于生成回答")
        
        # 6. 生成答案
        print("正在基于精选片段生成答案...")
        answer = self.generate_answer(user_input, final_docs)
        
        # 7. 记录本次交互
        self._log_interaction(user_input, answer)
        
        return answer