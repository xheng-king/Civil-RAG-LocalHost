# traindata_generator.py
import os
import json
import time
from typing import List, Dict, Tuple, Optional

import chromadb
from openai import OpenAI

# 导入配置
from settings import (
    embedding_API_key,
    base_url_set,
    embedding_model,
    BASE_INITIAL_RETRIEVE_K
)


class SwiftDataGenerator:
    def __init__(self):
        """初始化生成器，设置 embedding 客户端和 Chroma 客户端，但不指定集合"""
        if not embedding_API_key:
            raise ValueError("settings.py 中的 embedding_API_key 未设置")

        self.client = OpenAI(
            api_key=embedding_API_key,
            base_url=base_url_set
        )
        self.chroma_client = chromadb.PersistentClient(path="../data/vectorstore")
        self.collection_name = None
        self.collection = None
        self.retrieve_k = BASE_INITIAL_RETRIEVE_K

    def list_collections(self) -> List[str]:
        """列出 Chroma 中的所有集合名称"""
        try:
            collections = self.chroma_client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            print(f"获取集合列表失败: {e}")
            return []

    def select_collection_by_user(self) -> bool:
        """交互式选择集合，返回是否成功选择"""
        collections = self.list_collections()
        if not collections:
            print("错误: 当前没有可用的 Chroma 集合，请先运行索引器建立集合。")
            return False

        print("\n可用的 Chroma 集合:")
        for i, name in enumerate(collections, 1):
            try:
                col = self.chroma_client.get_collection(name)
                count = col.count()
                print(f"  {i}. {name} (文档数: {count})")
            except:
                print(f"  {i}. {name}")

        while True:
            try:
                choice = input(f"\n请选择用于检索负样本的集合 (1-{len(collections)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(collections):
                    self.collection_name = collections[idx]
                    self.collection = self.chroma_client.get_collection(self.collection_name)
                    print(f"已选择集合: '{self.collection_name}'")
                    return True
                else:
                    print(f"请输入 1 到 {len(collections)} 之间的数字")
            except ValueError:
                print("请输入有效的数字")

    def load_qa_pairs(self, file_path: str) -> List[Dict[str, str]]:
        """从问答文件（.json / .jsonl）中加载问答对"""
        qa_pairs = []
        print(f"正在加载文件: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return []

            try:
                if content.startswith('['):
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            if 'question' in item and 'answer' in item:
                                q = str(item['question']).strip()
                                a = str(item['answer']).strip()
                                if q and a:
                                    qa_pairs.append({'question': q, 'answer': a})
                else:
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            if 'question' in item and 'answer' in item:
                                q = str(item['question']).strip()
                                a = str(item['answer']).strip()
                                if q and a:
                                    qa_pairs.append({'question': q, 'answer': a})
                        except json.JSONDecodeError:
                            continue
            except json.JSONDecodeError:
                print("文件格式解析失败")
                return []

        print(f"  成功加载 {len(qa_pairs)} 个有效问答对")
        return qa_pairs

    def retrieve_documents_with_scores(self, query: str) -> List[Tuple[str, float]]:
        """
        对给定的 query 调用 embedding 模型，并在 Chroma 集合中检索最相似的文档。
        返回列表，每个元素为 (文档内容, 相似度标签)，相似度标签范围 (0,1]，值越大表示越相似。
        相似度由 Chroma 返回的 L2 距离转换得到：similarity = 1 / (1 + distance)
        """
        if self.collection is None:
            raise RuntimeError("未选择 Chroma 集合，无法检索")

        try:
            response = self.client.embeddings.create(
                model=embedding_model,
                input=query
            )
            query_embedding = response.data[0].embedding

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.retrieve_k,
                include=["documents", "distances"]
            )

            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []

            docs_with_scores = []
            for doc, dist in zip(documents, distances):
                similarity = 1.0 / (1.0 + dist)   # 平方欧氏距离 -> 相似度，范围 (0,1]
                # 保留三位小数
                similarity_rounded = round(similarity, 3)
                docs_with_scores.append((doc, similarity_rounded))

            return docs_with_scores

        except Exception as e:
            print(f"  检索失败: {e}，将返回空列表")
            return []

    def generate_pairs_for_file(
        self,
        file_path: str,
        output_dir: str,
        max_global_count: int,
        current_count: List[int]
    ) -> bool:
        """
        处理单个问答文件：对每个 query 生成 1 个正样本 (label=1.0) + K 个负样本（标签为检索相似度）。
        所有标签保留三位小数。
        """
        qa_pairs = self.load_qa_pairs(file_path)
        if len(qa_pairs) < 1:
            print("  问答对数量不足，跳过。")
            return False

        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{name_without_ext}_initial.jsonl")
        os.makedirs(output_dir, exist_ok=True)

        written_count = 0
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for idx, qa in enumerate(qa_pairs):
                if current_count[0] >= max_global_count:
                    return True

                query = qa['question']
                pos_doc = qa['answer']

                # ----- 1. 正样本 (label 保留三位小数：1.000) -----
                pos_pair = {
                    "query": query,
                    "response": pos_doc,
                    "label": 1.0   # 直接写 1.0，JSON 序列化时默认输出 1.0，非 1.000，但数值精度满足。
                               # 如需要严格 1.000 可改为 round(1.0, 3) 或 1.000，但 Python 浮点数表示不影响。
                }
                # 为保证输出美观，可显式格式化，但不影响训练。若要求文件内也显示三位小数，可以如下：
                # pos_pair["label"] = 1.000   # 但这会被读为浮点 1.0，仍然相同。
                # 为满足字面要求，我们使用格式化写入:
                line = json.dumps(pos_pair, ensure_ascii=False, default=round_decimal)
                f_out.write(line + "\n")
                current_count[0] += 1
                written_count += 1

                if current_count[0] >= max_global_count:
                    break

                # ----- 2. 负样本 -----
                print(f"  查询 [{idx+1}/{len(qa_pairs)}]: {query[:50]}...")
                docs_with_scores = self.retrieve_documents_with_scores(query)

                if not docs_with_scores:
                    print(f"    未检索到文档，跳过负样本生成")
                    continue

                for doc, sim_label in docs_with_scores:
                    if current_count[0] >= max_global_count:
                        break

                    if doc.strip() == pos_doc.strip():
                        continue

                    neg_pair = {
                        "query": query,
                        "response": doc,
                        "label": sim_label   # 已经是保留三位小数的数值
                    }
                    f_out.write(json.dumps(neg_pair, ensure_ascii=False) + "\n")
                    current_count[0] += 1
                    written_count += 1

        print(f"  本文件生成 {written_count} 条 Pair 数据 → {output_file}")
        return current_count[0] >= max_global_count

    def run(self):
        print("\n--- SWIFT Embedding 数据生成器 (检索式负样本，相似度作为标签，保留三位小数) ---")
        print(f"检索参数: BASE_INITIAL_RETRIEVE_K = {self.retrieve_k}")
        print("说明：负样本的 label 由 Chroma 返回的 L2 距离转换为相似度（1/(1+distance)），并保留三位小数。")

        if not self.select_collection_by_user():
            return

        while True:
            try:
                max_input = input("请输入期望生成的最大数据对总数 (默认 1000): ").strip()
                max_count = int(max_input) if max_input else 1000
                if max_count > 0:
                    break
                print("请输入正整数")
            except ValueError:
                print("请输入有效的数字")

        test_dir = "../data/test/"
        if not os.path.exists(test_dir):
            print(f"错误: 目录不存在: {test_dir}")
            return

        qa_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jsonl', '.json'))])
        if not qa_files:
            print("未找到问答文件 (.json 或 .jsonl)")
            return

        print("\n可用文件：")
        for i, f in enumerate(qa_files, 1):
            print(f"  {i}. {f}")

        selection = input("\n请选择要处理的文件序号 (如 1,2,3 或全部留空): ").strip()
        selected_indices = []
        if not selection:
            selected_indices = list(range(len(qa_files)))
            print("已选择所有文件")
        else:
            try:
                selected_indices = [int(x) - 1 for x in selection.replace(" ", "").split(",")]
                selected_indices = [i for i in selected_indices if 0 <= i < len(qa_files)]
                if not selected_indices:
                    print("没有选择有效的文件")
                    return
            except ValueError:
                print("输入格式错误")
                return

        output_dir = "../data/training"
        current_count = [0]
        start_time = time.time()

        for i in selected_indices:
            fname = qa_files[i]
            full_path = os.path.join(test_dir, fname)
            print(f"\n>>> 处理文件 [{i+1}/{len(selected_indices)}]: {fname}")

            stop_flag = self.generate_pairs_for_file(
                full_path,
                output_dir,
                max_count,
                current_count
            )
            if stop_flag:
                print("\n已达到最大生成数量限制，停止处理。")
                break

        end_time = time.time()
        print(f"\n✅ 全部完成！")
        print(f"📊 共生成 {current_count[0]} 条训练数据")
        print(f"📂 保存路径：{output_dir}")
        print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")


def round_decimal(obj):
    """用于json.dumps的default函数，将浮点数保留三位小数（可选，但此处未强制使用）"""
    if isinstance(obj, float):
        return round(obj, 3)
    raise TypeError


def main():
    try:
        generator = SwiftDataGenerator()
        generator.run()
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()