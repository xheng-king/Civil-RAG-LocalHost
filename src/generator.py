# generator.py
import time
import math
import re
import json
import csv
import os
import glob
from typing import List, Dict, Any, Optional, Tuple, Callable
from openai import OpenAI
from tqdm import tqdm
from retriever import Retriever
from settings import (
    ENABLE_ADAPTIVE_RETRIEVAL,
    MAX_RETRIEVAL_ROUNDS,
    RETRIEVAL_STEP_SIZE,
    RERANK_OUTPUT_STEP_SIZE,
    BASE_INITIAL_RETRIEVE_K,
    BASE_FINAL_TOP_K,
    llm, llm_base_url, llm_API_key
)


# ---------- 辅助评估函数（原 rag_interface 中的内容）----------
def extract_chinese_chars(text: str) -> str:
    """提取字符串中的所有中文字符"""
    return re.sub(r'[^\u4e00-\u9fff]', '', text)


def calculate_bleu_score(candidate: str, reference: str) -> float:
    """
    计算 BLEU 分数（仅中文字符，1-3 gram 均匀权重）
    需要 nltk 库，如果没有则返回 0.0 并给出警告
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        print("警告: nltk 未安装，BLEU 分数将返回 0.0")
        return 0.0

    candidate_zh = extract_chinese_chars(candidate)
    reference_zh = extract_chinese_chars(reference)

    if not candidate_zh or not reference_zh:
        return 0.0

    candidate_tokens = list(candidate_zh)
    reference_tokens = list(reference_zh)
    weights = (1/3, 1/3, 1/3)
    smoothing = SmoothingFunction().method1

    try:
        bleu = sentence_bleu([reference_tokens], candidate_tokens, weights=weights, smoothing_function=smoothing)
        return bleu
    except Exception as e:
        print(f"计算 BLEU 时出错: {e}")
        return 0.0


def check_answer_correctness(question: str, generated_answer: str, reference_answer: str) -> bool:
    """调用 LLM 判断生成答案是否正确"""
    prompt = f"""
    你是一个专业的评判员。
    你的任务是判断待评价答案是否与参考答案意义相近或结果相同。
    
    请严格只回复 "CORRECT" 或 "INCORRECT"。

    问题: {question}

    参考标准答案: {reference_answer}

    待评估答案: {generated_answer}
    """

    try:
        client = OpenAI(api_key=llm_API_key, base_url=llm_base_url)
        response = client.chat.completions.create(
            model=llm,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        result_text = response.choices[0].message.content.strip().upper()
        return result_text == "CORRECT"
    except Exception as e:
        print(f"调用大模型进行准确性评估时出错: {e}. 问题: '{question[:50]}...'. 视为不正确。")
        return False


# ---------- 生成器类 ----------
class Generator:
    """生成器：调用检索器获取文档，生成答案；支持自适应重试（基于查询自身的MRR）"""

    def __init__(self):
        self.retriever = Retriever()
        if not llm_API_key:
            raise ValueError("settings.py 中的 llm_API_key 未设置")
        self.llm_client = OpenAI(api_key=llm_API_key, base_url=llm_base_url)
        self.log_file_path = "../query_log.md"

        self.initial_retrieve_k = BASE_INITIAL_RETRIEVE_K
        self.final_top_k = BASE_FINAL_TOP_K
        self.collection_name = None

    # -------------------- 日志 --------------------
    def _log_interaction(self, user_input: str, response: str, round_num: int = 1, status: str = "Final"):
        markdown_content = f"--- Round {round_num} ({status}) ---\nQ：{user_input}\nA：{response}\n\n"
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(markdown_content)
        except Exception as e:
            print(f"记录日志时出错: {e}")

    # -------------------- 检索与重排序核心 --------------------
    def _retrieve_and_rerank(self, query: str, retrieve_k: int, top_k: int) -> Tuple[List[Dict[str, Any]], float, float]:
        self.retriever.initial_retrieve_k = retrieve_k
        self.retriever.final_top_k = top_k

        candidates = self.retriever.retrieve_documents(query, k=retrieve_k)
        if not candidates:
            return [], 0.0, 0.0

        all_reranked = self.retriever.rerank_documents(query, candidates, top_n=len(candidates))
        final_docs = all_reranked[:top_k]

        mrr = self.retriever.calc_mrr_by_coverage(final_docs, query, threshold=0.6)
        ndcg = self.retriever.calc_ndcg_by_coverage(final_docs, query)
        return final_docs, mrr, ndcg

    # -------------------- 答案生成 --------------------
    def _generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        if not contexts:
            return "抱歉，没有找到相关文档。"

        context_str = "\n\n".join([
            f"参考信息 #{doc.get('rerank_rank', i+1)} (相关性分数: {doc.get('rerank_score', 0):.4f}, 来源: {doc.get('metadata', {}).get('source', '未知')}):\n{doc['content']}"
            for i, doc in enumerate(contexts)
        ])

        prompt = f"""基于以下数据库内容，回答用户的问题。

数据库内容：
{context_str}

用户问题：
{query}

回答要求：
1. 若数据库内容中存在用户问题的相关回答，则直接简明扼要地回答问题
2. 若数据库中不存在用户问题的相关解答，提示用户查询结果没有相关内容
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

    # -------------------- 主查询入口 --------------------
    def query(self,
              user_input: str,
              evaluator_func: Optional[Callable[[str, str, str], bool]] = None,
              reference_answer: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        主查询接口，支持交互模式和评估模式。
        """
        if self.collection_name is None:
            self.collection_name = self.retriever.select_collection()
            if self.collection_name is None:
                raise RuntimeError("未选择有效集合，无法查询")

        if self.retriever.collection is None:
            self.retriever.collection = self.retriever.chroma_client.get_collection(self.collection_name)

        if not ENABLE_ADAPTIVE_RETRIEVAL:
            # ---------- 单次模式 ----------
            final_docs, mrr, ndcg = self._retrieve_and_rerank(user_input, self.initial_retrieve_k, self.final_top_k)
            answer = self._generate_answer(user_input, final_docs)
            self._log_interaction(user_input, answer, round_num=1, status="Standard")

            if reference_answer is not None and evaluator_func is not None:
                bleu = calculate_bleu_score(answer, reference_answer)
                acc = 1 if check_answer_correctness(user_input, answer, reference_answer) else 0
                metrics = {"mrr": mrr, "ndcg": ndcg, "bleu": bleu, "acc": acc, "retrieval_rounds": 1}
                return answer, metrics
            else:
                return answer, {}
        else:
            # ---------- 自适应模式 ----------
            current_k = self.initial_retrieve_k
            current_top_k = self.final_top_k
            ndcg = 0.0
            mrr = 0.0
            docs = []
            rounds = 0

            for round_num in range(1, MAX_RETRIEVAL_ROUNDS + 1):
                rounds = round_num
                docs, mrr, ndcg = self._retrieve_and_rerank(user_input, current_k, current_top_k)

                if mrr > 0:   # 只要有一个文档达到覆盖率阈值即停止
                    print(f"[自适应] 第 {round_num} 轮满足条件 (MRR={mrr:.4f})，停止重试")
                    break
                else:
                    if round_num < MAX_RETRIEVAL_ROUNDS:
                        current_k += RETRIEVAL_STEP_SIZE
                        current_top_k += RERANK_OUTPUT_STEP_SIZE
                        print(f"[自适应] 第 {round_num} 轮未达标 (MRR=0)，扩大参数: K={current_k}, TopN={current_top_k}")
                    else:
                        print(f"[自适应] 达到最大轮次 {MAX_RETRIEVAL_ROUNDS}")

            answer = self._generate_answer(user_input, docs)
            self._log_interaction(user_input, answer, round_num=rounds, status="Adaptive")

            if reference_answer is not None and evaluator_func is not None:
                bleu = calculate_bleu_score(answer, reference_answer)
                acc = 1 if check_answer_correctness(user_input, answer, reference_answer) else 0
                metrics = {
                    "mrr": mrr,
                    "ndcg": ndcg,
                    "bleu": bleu,
                    "acc": acc,
                    "retrieval_rounds": rounds
                }
                return answer, metrics
            else:
                return answer, {}

# ---------- 系统评测入口（供 main.py 调用） ----------
def run_evaluation():
    """系统评测入口（供 main.py 调用）- 支持断点续传，实时写入"""
    gen = Generator()
    
    # 1. 选择集合
    collection_name = gen.retriever.select_collection()
    if not collection_name:
        print("未选择集合，退出评估")
        return
    gen.collection_name = collection_name
    gen.retriever.collection = gen.retriever.chroma_client.get_collection(collection_name)
    
    # 2. 选择测试数据集
    test_dir = "../data/test"
    if not os.path.exists(test_dir):
        print(f"测试数据目录 {test_dir} 不存在")
        return
    
    jsonl_files = glob.glob(os.path.join(test_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"在 {test_dir} 中没有找到 .jsonl 文件")
        return
    
    print("\n可用的测试数据集:")
    for i, file_path in enumerate(jsonl_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    
    try:
        choice = input("\n请选择测试数据集 (输入序号，多个用逗号分隔，例如: 1 或 1,2,3): ").strip()
        if not choice:
            return
        indices = [int(i.strip()) - 1 for i in choice.split(',')]
        selected_files = [jsonl_files[idx] for idx in indices if 0 <= idx < len(jsonl_files)]
    except Exception as e:
        print(f"选择失败: {e}")
        return
    
    # 3. 准备结果保存路径
    result_dir = "../data/result"
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, "total.csv")
    jsonl_path = os.path.join(result_dir, "single_query.jsonl")
    
    # ----- 断点续传：读取已有记录并恢复全局指标列表 -----
    processed_queries = set()
    all_mrr, all_ndcg, all_bleu, all_acc, all_rounds = [], [], [], [], []
    
    if os.path.exists(jsonl_path):
        print(f"发现已有结果文件 {jsonl_path}，将跳过已评测的问题。")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    q = record["query"]
                    processed_queries.add(q)
                    all_mrr.append(record["mrr"])
                    all_ndcg.append(record["ndcg"])
                    all_bleu.append(record["bleu"])
                    all_acc.append(record["acc"])
                    all_rounds.append(record["retrieval_rounds"])
                except Exception as e:
                    print(f"解析已有记录失败: {e}")
        print(f"已加载 {len(processed_queries)} 条已有评测记录。")
    
    # ----- 以追加模式打开 JSONL，实时写入新结果 -----
    with open(jsonl_path, 'a', encoding='utf-8') as jsonl_file:
        for file_path in selected_files:
            print(f"\n{'='*60}\n评估数据集: {os.path.basename(file_path)}")
            test_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        test_data.append(item)
                    except:
                        continue
            if not test_data:
                continue
            
            # 用于统计本次运行中该文件新评测的问题（仅用于展示）
            new_mrr, new_ndcg, new_bleu, new_acc, new_rounds = [], [], [], [], []
            skipped_count = 0
            
            for item in tqdm(test_data, desc="处理问题"):
                query = item["question"]
                ref_answer = item["answer"]
                
                # 跳过已评测的 query
                if query in processed_queries:
                    skipped_count += 1
                    continue
                
                # 调用核心查询接口（包含检索和生成）
                answer, metrics = gen.query(
                    user_input=query,
                    evaluator_func=check_answer_correctness,
                    reference_answer=ref_answer
                )
                
                mrr = metrics.get("mrr", 0.0)
                ndcg = metrics.get("ndcg", 0.0)
                bleu = metrics.get("bleu", 0.0)
                acc = metrics.get("acc", 0)
                rounds = metrics.get("retrieval_rounds", 1)
                
                # 更新全局指标列表
                all_mrr.append(mrr)
                all_ndcg.append(ndcg)
                all_bleu.append(bleu)
                all_acc.append(acc)
                all_rounds.append(rounds)
                
                # 更新当前文件的新指标列表
                new_mrr.append(mrr)
                new_ndcg.append(ndcg)
                new_bleu.append(bleu)
                new_acc.append(acc)
                new_rounds.append(rounds)
                
                # 记录到 JSONL（每个问题立即写入）
                record = {
                    "query": query,
                    "reference_answer": ref_answer,
                    "generated_answer": answer,
                    "mrr": mrr,
                    "ndcg": ndcg,
                    "bleu": bleu,
                    "acc": acc,
                    "retrieval_rounds": rounds
                }
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                jsonl_file.flush()          # 确保立即写入磁盘
                
                # 将该 query 加入已处理集合，避免同一会话内重复（尽管已有跳过逻辑）
                processed_queries.add(query)
            
            # 输出当前数据集的本次运行结果（仅针对新评测的问题）
            if new_mrr:
                avg_mrr = sum(new_mrr) / len(new_mrr)
                avg_ndcg = sum(new_ndcg) / len(new_ndcg)
                avg_bleu = sum(new_bleu) / len(new_bleu)
                avg_acc = sum(new_acc) / len(new_acc)
                avg_rounds = sum(new_rounds) / len(new_rounds)
                print(f"\n本次新增评测 {len(new_mrr)} 条，数据集结果: MRR={avg_mrr:.4f}, NDCG={avg_ndcg:.4f}, BLEU={avg_bleu:.4f}, ACC={avg_acc:.4f}, 平均轮次={avg_rounds:.2f}")
            else:
                print(f"该数据集所有问题均已评测过，无新增。")
            if skipped_count > 0:
                print(f"（已跳过 {skipped_count} 个之前评测过的问题）")
    
    # ----- 总体结果（包含历史+新增）-----
    if all_mrr:
        overall = {
            "MRR": sum(all_mrr) / len(all_mrr),
            "NDCG": sum(all_ndcg) / len(all_ndcg),
            "BLEU": sum(all_bleu) / len(all_bleu),
            "ACC": sum(all_acc) / len(all_acc),
            "Avg_Rounds": sum(all_rounds) / len(all_rounds)
        }
        print(f"\n{'='*60}\n总体评估结果（所有已评测问题）:")
        for k, v in overall.items():
            print(f"  {k}: {v:.4f}")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in overall.items():
                writer.writerow([k, f"{v:.4f}"])
        print(f"\n结果已保存至: {os.path.abspath(csv_path)} 和 {os.path.abspath(jsonl_path)}")
    else:
        print("没有有效的评估数据。")
        
# ---------- 交互式入口（供 main.py 调用） ----------
def run_interactive():
    """交互式问答入口（供 main.py 调用）"""
    gen = Generator()
    
    # 1. 选择集合
    collection_name = gen.retriever.select_collection()
    if not collection_name:
        print("未选择集合，退出交互系统")
        return
    gen.collection_name = collection_name
    gen.retriever.collection = gen.retriever.chroma_client.get_collection(collection_name)
    
    # 2. 打印模式信息
    print("=" * 60)
    print("交互式问答系统")
    print("=" * 60)
    mode_str = "自适应重试模式" if ENABLE_ADAPTIVE_RETRIEVAL else "标准模式"
    print(f"当前模式: {mode_str}")
    print(f"基础初始召回: {gen.initial_retrieve_k} | 基础重排序后使用: {gen.final_top_k}")
    if ENABLE_ADAPTIVE_RETRIEVAL:
        print(f"重试策略: 每次失败 K+{RETRIEVAL_STEP_SIZE}, TopN+{RERANK_OUTPUT_STEP_SIZE}, 最大轮次 {MAX_RETRIEVAL_ROUNDS}")
    print(f"查询日志保存至: {gen.log_file_path}")
    print("输入 'quit' 或 'exit' 退出系统\n")
    
    # 3. 交互循环
    while True:
        try:
            user_input = input("请输入您的问题: ").strip()
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            if not user_input:
                continue
            
            start_time = time.time()
            answer, _ = gen.query(user_input)   # 交互模式，不传入评估参数
            elapsed = time.time() - start_time
            
            print("\n" + "=" * 40)
            print(f"回答 (处理时间: {elapsed:.2f}秒):")
            print(answer)
            print("=" * 40 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    # 可直接运行交互式
    run_interactive()