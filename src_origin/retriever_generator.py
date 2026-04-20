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
# --- 导入jieba用于中文分词 ---
import jieba
# --- 导入NLTK BLEU相关模块 ---
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore') # 忽略NLTK可能产生的警告

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

def select_test_datasets():
    """让用户选择测试数据集"""
    test_dir = "../data/test"
    if not os.path.exists(test_dir):
        print(f"测试数据目录 {test_dir} 不存在")
        return None
    
    # 获取所有 .jsonl 文件
    jsonl_files = glob.glob(os.path.join(test_dir, "*.jsonl"))
    
    if not jsonl_files:
        print(f"在 {test_dir} 中没有找到 .jsonl 文件")
        return None
    
    print("\n可用的测试数据集:")
    for i, file_path in enumerate(jsonl_files, 1):
        file_name = os.path.basename(file_path)
        print(f"  {i}. {file_name}")
    
    while True:
        try:
            choice = input("\n请选择测试数据集 (输入序号，多个用逗号分隔，例如: 1 或 1,2,3): ").strip()
            if not choice:
                continue
                
            if choice.lower() in ['quit', 'exit']:
                return None
                
            # 解析选择
            indices = [int(i.strip()) - 1 for i in choice.split(',')]
            
            # 检查索引是否有效
            if all(0 <= idx < len(jsonl_files) for idx in indices):
                selected_files = [jsonl_files[idx] for idx in indices]
                return selected_files
            else:
                print(f"请输入 1 到 {len(jsonl_files)} 之间的有效序号")
        except ValueError:
            print("请输入有效的数字序号，多个序号用逗号分隔")
        except Exception as e:
            print(f"选择数据集时出错: {e}")
            return None

def load_test_data(file_path):
    """加载测试数据集"""
    test_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    test_data.append(item)
                except json.JSONDecodeError:
                    print(f"跳过无效的JSON行: {line}")
        print(f"已加载 {len(test_data)} 个测试问题 from {os.path.basename(file_path)}")
        return test_data
    except Exception as e:
        print(f"加载测试数据时出错: {e}")
        return []

def calculate_dcg(grades):
    """计算 DCG"""
    dcg = 0
    for i, grade in enumerate(grades):
        dcg += (2**grade - 1) / math.log2(i + 2)  # i+2 因为排名从1开始，log2(rank+1)
    return dcg

def calculate_idcg(grades):
    """计算 IDCG（理想情况下的 DCG）"""
    # 按相关性等级降序排列
    sorted_grades = sorted(grades, reverse=True)
    return calculate_dcg(sorted_grades)

def get_ngrams(text, n):
    """获取文本的所有n-gram"""
    tokens = text.split()
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams

def calculate_bleu_score(candidate, reference, max_n=4):
    """
    使用 NLTK 计算 BLEU 得分
    :param candidate: 生成的句子 (str)
    :param reference: 参考句子 (str)
    :param max_n: 最大的 n-gram 阶数 (默认 4)
    :return: BLEU 得分 (float)
    """
    if not candidate or not reference:
        return 0.0

    try:
        # 使用 jieba 进行中文分词
        # jieba.lcut 返回一个分词后的列表
        candidate_tokens = jieba.lcut(candidate.strip())
        reference_tokens = jieba.lcut(reference.strip())

        print(f"DEBUG: Candidate Tokens: {candidate_tokens}")
        print(f"DEBUG: Reference Tokens: {reference_tokens}")

        # 检查分词结果是否为列表
        if not isinstance(candidate_tokens, list) or not isinstance(reference_tokens, list):
            print(f"分词结果类型错误: candidate type={type(candidate_tokens)}, reference type={type(reference_tokens)}")
            return 0.0

        # 如果候选翻译的 token 数量为 0，则无法计算 BLEU
        if len(candidate_tokens) == 0:
            return 0.0

        # 定义权重，实现等权重的几何平均
        weights = (0.25, 0.25, 0.25, 0.25)

        # 使用 smoothing function 来处理零精度问题
        chencherry = SmoothingFunction()
        
        # 计算 BLEU 分数
        # 确保 references 是一个包含列表的列表
        references_for_nltk = [reference_tokens]
        
        # 确保 hypothesis 是一个列表
        hypothesis_for_nltk = candidate_tokens
        
        bleu_score = sentence_bleu(
            references=references_for_nltk,
            hypothesis=hypothesis_for_nltk,
            weights=weights,
            smoothing_function=chencherry.method1
        )

        return bleu_score

    except Exception as e:
        # 捕获更具体的错误信息
        import traceback
        print(f"计算 BLEU 时发生异常: {e}")
        traceback.print_exc() # 打印完整的堆栈跟踪
        print(f"  Candidate: '{candidate}', Reference: '{reference}'")
        return 0.0

def evaluate_from_test_data():
    """从测试数据评估 RAG 系统的性能指标"""
    try:
        rag_system = QwenRetrieverGenerator()
        
        # 让用户选择集合
        collection_name = rag_system.select_collection()
        if not collection_name:
            print("无法选择集合，退出评估")
            return
        
        # 选择测试数据集
        test_files = select_test_datasets()
        if not test_files:
            print("未选择测试数据集，退出评估")
            return
        
        # 准备结果存储
        all_mrr_initial = []
        all_ndcg_initial = []
        all_mrr_reranked = []
        all_ndcg_reranked = []
        all_bleu = []  # 新增 BLEU 得分列表
        
        # 遍历每个测试数据集
        for file_path in test_files:
            print(f"\n{'='*60}")
            print(f"开始评估数据集: {os.path.basename(file_path)}")
            print(f"{'='*60}")
            
            # 加载测试数据
            test_data = load_test_data(file_path)
            if not test_data:
                continue
            
            # 存储当前数据集的指标
            mrr_initial_scores = []
            ndcg_initial_scores = []
            mrr_reranked_scores = []
            ndcg_reranked_scores = []
            bleu_scores = []  # 新增 BLEU 得分列表
            
            # 遍历每个问题
            for i, item in enumerate(tqdm(test_data, desc="处理问题")):
                query = item["question"]
                reference_answer = item["answer"]  # 获取标准答案
                
                # 1. 检索候选文档 (初始召回)
                candidate_docs = rag_system.retrieve_documents(query)
                
                if not candidate_docs:
                    print(f"  问题 #{i+1}: 未检索到任何文档")
                    # 对于初始召回
                    mrr_initial_scores.append(0.1)  # 1/10 = 0.1
                    ndcg_initial_scores.append(0)
                    # 对于重排序后
                    mrr_reranked_scores.append(0.1)
                    ndcg_reranked_scores.append(0)
                    # BLEU
                    bleu_scores.append(0)  # 无生成答案，BLEU为0
                    continue
                
                # --- 计算 MRR 和 NDCG (基于初始召回的 candidate_docs) ---
                # 2. 计算 MRR - 找到第一个相关文档的排名 (基于初始召回结果)
                first_relevant_rank_initial = rag_system.get_relevance_rank(query, candidate_docs)
                mrr_initial = 1.0 / first_relevant_rank_initial if first_relevant_rank_initial <= 5 else 0.1
                mrr_initial_scores.append(mrr_initial)
                
                # 3. 计算 NDCG (基于初始召回的 candidate_docs)
                # 先获取每个文档的相关性等级 (基于初始召回结果)
                grades_initial = []
                for doc in candidate_docs: # 使用初始召回的 candidate_docs
                    grade = rag_system.get_relevance_grade(query, doc)
                    grades_initial.append(grade)
                
                # 计算 DCG 和 IDCG
                dcg_initial = calculate_dcg(grades_initial)
                idcg_initial = calculate_idcg(grades_initial)
                
                # 计算 NDCG
                ndcg_initial = dcg_initial / idcg_initial if idcg_initial > 0 else 0
                ndcg_initial_scores.append(ndcg_initial)
                # --- 初始召回的 MRR 和 NDCG 计算结束 ---
                
                # --- 对初始召回的文档进行重排序 ---
                reranked_docs = rag_system.rerank_documents(query, candidate_docs)
                
                # --- 计算 MRR 和 NDCG (基于重排序后的 reranked_docs) ---
                # 4. 计算重排序后的 MRR - 找到第一个相关文档的排名 (基于重排序结果)
                first_relevant_rank_reranked = rag_system.get_relevance_rank(query, reranked_docs)
                mrr_reranked = 1.0 / first_relevant_rank_reranked if first_relevant_rank_reranked <= 5 else 0.1
                mrr_reranked_scores.append(mrr_reranked)
                
                # 5. 计算重排序后的 NDCG (基于重排序后的 reranked_docs)
                # 先获取每个文档的相关性等级 (基于重排序结果)
                grades_reranked = []
                for doc in reranked_docs: # 使用重排序后的 reranked_docs
                    grade = rag_system.get_relevance_grade(query, doc)
                    grades_reranked.append(grade)
                
                # 计算 DCG 和 IDCG (基于重排序后的 grades)
                dcg_reranked = calculate_dcg(grades_reranked)
                idcg_reranked = calculate_idcg(grades_reranked) # 使用重排序后的 grades 计算 IDCG
                
                # 计算 NDCG
                ndcg_reranked = dcg_reranked / idcg_reranked if idcg_reranked > 0 else 0
                ndcg_reranked_scores.append(ndcg_reranked)
                # --- 重排序后的 MRR 和 NDCG 计算结束 ---
                
                # 6. 生成答案并计算 BLEU (使用重排序后的文档)
                final_docs_for_generation = reranked_docs[:rag_system.final_top_k]
                generated_answer = rag_system.generate_answer(query, final_docs_for_generation) # 这里会内部使用 rerank 信息

                print(f"DEBUG: Generated Answer: '{generated_answer}'") # 打印生成的答案
                print(f"DEBUG: Reference Answer: '{reference_answer}'") # 打印参考答案

                # 计算 BLEU 得分
                bleu_score = calculate_bleu_score(generated_answer, reference_answer)
                bleu_scores.append(bleu_score)
                
                # 打印当前问题的部分结果
                print(f"  问题 #{i+1}: MRR_Init={mrr_initial:.4f}, NDCG_Init={ndcg_initial:.4f}, MRR_Rerank={mrr_reranked:.4f}, NDCG_Rerank={ndcg_reranked:.4f}, BLEU={bleu_score:.4f}")
            
            # 计算数据集的平均指标
            dataset_mrr_initial = sum(mrr_initial_scores) / len(mrr_initial_scores) if mrr_initial_scores else 0
            dataset_ndcg_initial = sum(ndcg_initial_scores) / len(ndcg_initial_scores) if ndcg_initial_scores else 0
            dataset_mrr_reranked = sum(mrr_reranked_scores) / len(mrr_reranked_scores) if mrr_reranked_scores else 0
            dataset_ndcg_reranked = sum(ndcg_reranked_scores) / len(ndcg_reranked_scores) if ndcg_reranked_scores else 0
            dataset_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0  # 计算平均BLEU
            
            print(f"\n数据集 {os.path.basename(file_path)} 评估结果:")
            print(f"  MRR_Initial: {dataset_mrr_initial:.4f}")
            print(f"  NDCG_Initial: {dataset_ndcg_initial:.4f}")
            print(f"  MRR_Reranked: {dataset_mrr_reranked:.4f}")
            print(f"  NDCG_Reranked: {dataset_ndcg_reranked:.4f}")
            print(f"  BLEU: {dataset_bleu:.4f}")  # 打印BLEU结果
            
            # 存储到总结果
            all_mrr_initial.extend(mrr_initial_scores)
            all_ndcg_initial.extend(ndcg_initial_scores)
            all_mrr_reranked.extend(mrr_reranked_scores)
            all_ndcg_reranked.extend(ndcg_reranked_scores)
            all_bleu.extend(bleu_scores)  # 添加BLEU得分到总列表
        
        # 计算总体指标
        overall_mrr_initial = sum(all_mrr_initial) / len(all_mrr_initial) if all_mrr_initial else 0
        overall_ndcg_initial = sum(all_ndcg_initial) / len(all_ndcg_initial) if all_ndcg_initial else 0
        overall_mrr_reranked = sum(all_mrr_reranked) / len(all_mrr_reranked) if all_mrr_reranked else 0
        overall_ndcg_reranked = sum(all_ndcg_reranked) / len(all_ndcg_reranked) if all_ndcg_reranked else 0
        overall_bleu = sum(all_bleu) / len(all_bleu) if all_bleu else 0  # 计算总体BLEU
        
        print(f"\n{'='*60}")
        print("总体评估结果:")
        print(f"  MRR_Initial: {overall_mrr_initial:.4f}")
        print(f"  NDCG_Initial: {overall_ndcg_initial:.4f}")
        print(f"  MRR_Reranked: {overall_mrr_reranked:.4f}")
        print(f"  NDCG_Reranked: {overall_ndcg_reranked:.4f}")
        print(f"  BLEU: {overall_bleu:.4f}")  # 打印总体BLEU
        print(f"{'='*60}")
        
        # 保存结果到 CSV
        result_file = "../evaluation_results.csv"
        with open(result_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["MRR_Initial", f"{overall_mrr_initial:.4f}"])
            writer.writerow(["NDCG_Initial", f"{overall_ndcg_initial:.4f}"])
            writer.writerow(["MRR_Reranked", f"{overall_mrr_reranked:.4f}"])
            writer.writerow(["NDCG_Reranked", f"{overall_ndcg_reranked:.4f}"])
            writer.writerow(["BLEU", f"{overall_bleu:.4f}"])  # 写入BLEU结果
        
        print(f"\n评估结果已保存到: {os.path.abspath(result_file)}")
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc() # 添加更详细的错误追踪

def interactive_query():
    """交互式查询"""
    try:
        rag_system = QwenRetrieverGenerator()
        
        # 让用户选择集合
        collection_name = rag_system.select_collection()
        if not collection_name:
            print("无法选择集合，退出查询")
            return
        
        print("="*60)
        print("RAG 问答系统")
        print("="*60)
        print(f"初始召回: {rag_system.initial_retrieve_k} 个片段 | 重排序后使用: {rag_system.final_top_k} 个片段")
        print(f"查询日志将保存至: {os.path.abspath(rag_system.log_file_path)}")
        print("输入 'quit' 或 'exit' 退出系统")
        print()
        
        while True:
            try:
                # 从标准输入获取用户输入
                user_input = input("请输入您的问题: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                if not user_input:
                    continue
                
                # 执行查询
                start_time = time.time()
                response = rag_system.query(user_input)
                elapsed = time.time() - start_time
                
                # 打印结果到标准输出
                print("\n" + "="*40)
                print(f"回答 (处理时间: {elapsed:.2f}秒):")
                print(response)
                print("="*40 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                
    except ValueError as e:
        print(f"错误: {e}")
        print("请确保已设置 OPENAI_API_KEY 环境变量")
    except Exception as e:
        print(f"启动系统时出错: {e}")

def main():
    """主函数，让用户选择模式"""
    print("="*60)
    print("RAG 系统")
    print("="*60)
    print("1. 交互式查询")
    print("2. 系统评估")
    print("3. 退出")
    
    while True:
        try:
            choice = input("\n请选择模式 (1-3): ").strip()
            
            if choice == '1':
                interactive_query()
                break
            elif choice == '2':
                evaluate_from_test_data()
                break
            elif choice == '3' or choice.lower() in ['quit', 'exit']:
                print("再见！")
                break
            else:
                print("请输入有效的选项 (1-3)")
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    main()