# retriever.py
import os
import math
import requests
import chromadb
from openai import OpenAI
from typing import List, Dict, Any, Optional
from database_manager import DatabaseManager
from settings import (
    base_url_set, embedding_model, embedding_API_key,
    rerank_model, rerank_base_url, rerank_API_key,
    llm, llm_base_url, llm_API_key,
    BASE_INITIAL_RETRIEVE_K, BASE_FINAL_TOP_K
)


class Retriever:
    """检索器：负责向量检索、重排序、相关性评估及检索质量指标计算"""

    def __init__(self):
        # Embedding client
        if not embedding_API_key:
            raise ValueError("settings.py 中的 embedding_API_key 未设置")
        self.embedding_client = OpenAI(
            api_key=embedding_API_key,
            base_url=base_url_set
        )

        # LLM client（用于相关性评估）
        if not llm_API_key:
            raise ValueError("settings.py 中的 llm_API_key 未设置")
        self.llm_client = OpenAI(
            api_key=llm_API_key,
            base_url=llm_base_url
        )

        # 重排序 API key
        self.rerank_api_key = rerank_API_key
        self.rerank_model = rerank_model
        self.rerank_base_url = rerank_base_url

        # ChromaDB 客户端
        self.chroma_client = chromadb.PersistentClient(path="../data/vectorstore")
        self.db_manager = DatabaseManager()

        # 检索参数默认值
        self.initial_retrieve_k = BASE_INITIAL_RETRIEVE_K
        self.final_top_k = BASE_FINAL_TOP_K

        self.collection = None  # 当前选中的集合

    # -------------------- 集合管理 --------------------
    def select_collection(self) -> Optional[str]:
        """交互式选择要查询的集合，返回集合名称"""
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

    # -------------------- 向量化 --------------------
    def embed_query(self, query_text: str) -> List[float]:
        """将查询文本转换为向量"""
        response = self.embedding_client.embeddings.create(
            model=embedding_model,
            input=query_text
        )
        return response.data[0].embedding

    # -------------------- 检索 --------------------
    def retrieve_documents(self, query_text: str, k: int = None) -> List[Dict[str, Any]]:
        """
        向量检索文档
        Args:
            query_text: 查询文本
            k: 返回的文档数量，默认使用 self.initial_retrieve_k
        Returns:
            文档列表，每个文档包含 content, metadata, initial_distance, score
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
                'score': 1.0 - dist,          # 余弦相似度
                'rerank_score': None
            })

        return retrieved_docs

    # -------------------- 重排序 --------------------
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]], top_n: int = None) -> List[Dict[str, Any]]:
        if top_n is None:
            top_n = self.final_top_k
        if not documents:
            return []
        actual_top_n = min(top_n, len(documents))
        if actual_top_n <= 0:
            return []

        try:
            headers = {"Authorization": f"Bearer {self.rerank_api_key}", "Content-Type": "application/json"}
            texts = [doc['content'] for doc in documents]
            payload = {"model": self.rerank_model, "documents": texts, "query": query, "top_n": len(documents)}
            response = requests.post(self.rerank_base_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if 'results' in result:
                reranked = []
                for rank_data in result['results']:
                    idx = rank_data['index']
                    doc = documents[idx].copy()
                    doc['rerank_score'] = rank_data['relevance_score']
                    reranked.append(doc)
                return reranked[:actual_top_n]
        except Exception as e:
            print(f"重排序失败: {e}")
        # 降级
        documents.sort(key=lambda x: 1.0 / (x['initial_distance'] + 1e-4), reverse=True)
        return documents[:actual_top_n]

    # -------------------- 检索质量指标（基于字符覆盖率） --------------------
    @staticmethod
    def extract_chinese_chars(text: str) -> str:
        """提取字符串中的所有中文字符（Unicode 4E00-9FFF）"""
        import re
        return re.sub(r'[^\u4e00-\u9fff]', '', text)

    @staticmethod
    def char_coverage(doc_text: str, ref_text: str) -> float:
        """
        计算文档中出现的参考答案中文字符所占比例（基于字符集合的交集）
        返回覆盖率（0~1）
        """
        doc_chinese = set(Retriever.extract_chinese_chars(doc_text))
        ref_chinese = set(Retriever.extract_chinese_chars(ref_text))
        if not ref_chinese:
            return 0.0
        common = doc_chinese.intersection(ref_chinese)
        return len(common) / len(ref_chinese)

    @staticmethod
    def is_document_relevant_by_coverage(doc_content: str, ref_answer: str, threshold: float = 0.6) -> bool:
        """根据字符覆盖率判断文档是否相关"""
        cov = Retriever.char_coverage(doc_content, ref_answer)
        return cov >= threshold

    def calc_mrr_by_coverage(self, reranked_docs: List[Dict[str, Any]], query: str, threshold: float = 0.6) -> float:
        """
        基于覆盖率计算 MRR：
        找到第一个覆盖率 >= threshold 的文档，返回其排名的倒数；若无则返回 0.0
        """
        for rank, doc in enumerate(reranked_docs, start=1):
            if self.is_document_relevant_by_coverage(doc.get('content', ''), query, threshold):
                return 1.0 / rank
        return 0.0

    def calc_ndcg_by_coverage(self, reranked_docs: List[Dict[str, Any]], query: str) -> float:
        """
        基于覆盖率计算 NDCG：
        使用每个文档的覆盖率作为连续相关性分数
        """
        if not reranked_docs:
            return 0.0

        scores = [self.char_coverage(doc.get('content', ''), query) for doc in reranked_docs]

        # 计算 DCG
        dcg = 0.0
        for i, score in enumerate(scores):
            gain = 2 ** score - 1
            dcg += gain / math.log2(i + 2)

        # 计算 IDCG
        ideal_scores = sorted(scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            gain = 2 ** score - 1
            idcg += gain / math.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0