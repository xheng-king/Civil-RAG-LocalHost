import os
import json
import random
import chromadb
from openai import OpenAI
from typing import List, Dict, Any
from database_manager import DatabaseManager
from settings import base_url_set, embedding_model, embedding_API_key, rerank_model, rerank_base_url, rerank_API_key


class TrainingDataGenerator:
    def __init__(self):
        if not embedding_API_key:
            raise ValueError("settings.py 中的 embedding_API_key 未设置")
        if not rerank_API_key:
            raise ValueError("settings.py 中的 rerank_API_key 未设置")

        self.embedding_client = OpenAI(api_key=embedding_API_key, base_url=base_url_set)
        self.chroma_client = chromadb.PersistentClient(path="../data/vectorstore")
        self.db_manager = DatabaseManager()
        self.collection = None
        self.initial_retrieve_k = 5

    def select_collection(self) -> bool:
        collection_names = self.db_manager.list_collections()
        if not collection_names:
            print("没有可用的集合，请先创建或索引一些数据")
            return False

        print("\n请选择要查询的集合:")
        for i, name in enumerate(collection_names, 1):
            print(f"  {i}. {name}")

        while True:
            try:
                choice = int(input(f"\n请选择 (1-{len(collection_names)}): "))
                if 1 <= choice <= len(collection_names):
                    selected_name = collection_names[choice - 1]
                    self.collection = self.chroma_client.get_collection(name=selected_name)
                    print(f"已选择集合: {selected_name}")
                    return True
                else:
                    print(f"请输入 1 到 {len(collection_names)} 之间的数字")
            except ValueError:
                print("请输入有效的数字")
            except EOFError:
                print("\n操作取消")
                return False

    def embed_query(self, query_text: str) -> List[float]:
        response = self.embedding_client.embeddings.create(model=embedding_model, input=query_text)
        return response.data[0].embedding

    def retrieve_documents(self, query_text: str) -> List[Dict[str, Any]]:
        query_embedding = self.embed_query(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.initial_retrieve_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        documents = results['documents'][0] or []
        metadatas = results['metadatas'][0] or []
        distances = results['distances'][0] or []

        retrieved = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            retrieved.append({
                'id': i,
                'content': doc,
                'metadata': meta,
                'initial_distance': dist,
                'rerank_score': None
            })
        return retrieved

    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {rerank_API_key}",
                "Content-Type": "application/json"
            }
            texts = [doc['content'] for doc in documents]
            payload = {
                "model": rerank_model,
                "documents": texts,
                "query": query,
                "top_n": len(documents)
            }
            response = requests.post(rerank_base_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            if 'results' in result:
                reranked = []
                for rank_data in result['results']:
                    idx = rank_data['index']
                    score = rank_data['relevance_score']
                    if idx < len(documents):
                        doc = documents[idx].copy()
                        doc['rerank_score'] = score
                        reranked.append(doc)
                reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
                return reranked
        except Exception as e:
            print(f"  重排序失败: {e}")
        return documents

    def process_qa_file(self, file_path: str, output_dir: str, max_triplets: int, total_generated: List[int]) -> bool:
        print(f"\n处理文件: {file_path}")
        qa_pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    qa = json.loads(line)
                    if 'question' in qa and 'answer' in qa:
                        qa_pairs.append(qa)
                except json.JSONDecodeError as e:
                    print(f"  行 {line_num} JSON 解析错误: {e}，跳过")

        if not qa_pairs:
            print(f"  无有效问答对")
            return False

        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{name_without_ext}_training.jsonl")
        os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f_out:
            file_written = 0
            for idx, qa in enumerate(qa_pairs, 1):
                if total_generated[0] >= max_triplets:
                    break

                question = qa['question']
                print(f"  处理 {idx}/{len(qa_pairs)}: {question[:50]}...")

                candidates = self.retrieve_documents(question)
                if len(candidates) < 2:
                    print(f"    文档不足，跳过")
                    continue

                reranked = self.rerank_documents(question, candidates)
                if not reranked:
                    continue

                pos_doc = reranked[0]['content']
                neg_candidates = [doc['content'] for doc in reranked[1:]]
                neg_doc = random.choice(neg_candidates)

                line = {
                    "query": question,
                    "pos": [pos_doc],
                    "neg": [neg_doc]
                }
                f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
                total_generated[0] += 1
                file_written += 1

        print(f"  本文件生成 {file_written} 条 → {output_file}")
        return total_generated[0] >= max_triplets

    def run(self):
        if not self.select_collection():
            return

        print("\n--- 参数设置 ---")
        while True:
            try:
                max_triplets = int(input("最大三元组个数: ").strip())
                if max_triplets > 0:
                    break
            except ValueError:
                print("请输入正整数")

        test_dir = "../data/test/"
        if not os.path.exists(test_dir):
            print(f"目录不存在: {test_dir}")
            return

        qa_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jsonl', '.json'))])
        if not qa_files:
            print("未找到问答文件")
            return

        print("\n可用文件：")
        for i, f in enumerate(qa_files, 1):
            print(f"  {i}. {f}")

        selection = input("\n选择序号（如 1,2,3）：").strip()
        try:
            selected = [int(x) - 1 for x in selection.replace(" ", "").split(",")]
            selected = [i for i in selected if 0 <= i < len(qa_files)]
        except:
            print("输入错误")
            return

        output_dir = "../data/training"
        total_generated = [0]
        for i in selected:
            fname = qa_files[i]
            full_path = os.path.join(test_dir, fname)
            self.process_qa_file(full_path, output_dir, max_triplets, total_generated)
            if total_generated[0] >= max_triplets:
                break

        print(f"\n✅ 全部完成！共生成 {total_generated[0]} 条 JSONL 训练数据")
        print(f"📂 路径：../data/training/xxx_training.jsonl")


def main():
    try:
        generator = TrainingDataGenerator()
        generator.run()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()