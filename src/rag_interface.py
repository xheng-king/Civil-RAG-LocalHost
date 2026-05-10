# rag_interface.py
import os
import re
import chromadb
from openai import OpenAI
from typing import List, Dict, Any
from database_manager import DatabaseManager
from retriever_generator import RetrieverGenerator
import time
import json
import csv
import glob
import math
from tqdm import tqdm
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

from settings import llm, llm_base_url, llm_API_key, ENABLE_ADAPTIVE_RETRIEVAL

eval_client = OpenAI(base_url=llm_base_url, api_key=llm_API_key)

def select_test_datasets():
    test_dir = "../data/test"
    if not os.path.exists(test_dir):
        print(f"测试数据目录 {test_dir} 不存在")
        return None
    
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
                
            indices = [int(i.strip()) - 1 for i in choice.split(',')]
            
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

def calc_mrr(scores):
    """根据分数列表计算 MRR（最高分文档视为唯一相关文档）[已弃用，保留兼容]"""
    if not scores:
        return 0.1
    max_idx = max(range(len(scores)), key=lambda i: scores[i])
    rank = max_idx + 1
    return 1.0 / rank if rank <= 5 else 0.1

def extract_chinese_chars(text: str) -> str:
    """提取字符串中的所有中文字符（Unicode 4E00-9FFF）"""
    return re.sub(r'[^\u4e00-\u9fff]', '', text)

def char_coverage(doc_text: str, ref_text: str) -> float:
    """
    计算文档中出现的参考答案中文字符所占比例（基于字符集合的交集）
    返回覆盖率（0~1）
    """
    doc_chinese = set(extract_chinese_chars(doc_text))
    ref_chinese = set(extract_chinese_chars(ref_text))
    if not ref_chinese:
        return 0.0
    common = doc_chinese.intersection(ref_chinese)
    return len(common) / len(ref_chinese)

def is_document_relevant_by_coverage(doc_content: str, ref_answer: str, threshold: float = 0.6) -> bool:
    """根据字符覆盖率判断文档是否相关（仅用于MRR的阈值判定）"""
    cov = char_coverage(doc_content, ref_answer)
    print(cov)  # 可选调试输出
    return cov >= threshold

def calc_mrr_by_coverage(reranked_docs: List[Dict], ref_answer: str, threshold: float = 0.6) -> float:
    """
    基于字符覆盖率计算的 MRR：
    在重排序后的文档列表中，找到第一个相关文档（覆盖率 >= threshold）的排名，返回倒数排名。
    若无相关文档，返回 0.0。
    """
    for rank, doc in enumerate(reranked_docs, start=1):
        if is_document_relevant_by_coverage(doc.get('content', ''), ref_answer, threshold):
            return 1.0 / rank
    return 0.0

def calc_ndcg_by_coverage(reranked_docs: List[Dict], ref_answer: str) -> float:
    """
    基于字符覆盖率计算的 NDCG：
    使用每个文档的覆盖率作为相关性分数（连续值），计算 NDCG。
    reranked_docs: 已按重排序模型排序的文档列表（顺序即为评估顺序）
    """
    if not reranked_docs:
        return 0.0
    
    dcg = 0.0
    for i, doc in enumerate(reranked_docs):
        score = char_coverage(doc.get('content', ''), ref_answer)
        gain = 2 ** score - 1
        dcg += gain / math.log2(i + 2)
    
    scores = [char_coverage(doc.get('content', ''), ref_answer) for doc in reranked_docs]
    ideal_scores = sorted(scores, reverse=True)
    idcg = 0.0
    for i, score in enumerate(ideal_scores):
        gain = 2 ** score - 1
        idcg += gain / math.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

# 保留原 calc_ndcg 函数（不再使用，仅兼容）
def calculate_dcg_from_scores(scores):
    dcg = 0.0
    for i, score in enumerate(scores):
        gain = 2 ** score - 1
        dcg += gain / math.log2(i + 2)
    return dcg

def calc_ndcg(scores):
    if not scores:
        return 0.0
    dcg = calculate_dcg_from_scores(scores)
    ideal_scores = sorted(scores, reverse=True)
    idcg = calculate_dcg_from_scores(ideal_scores)
    return dcg / idcg if idcg > 0 else 0.0

def calculate_bleu_score(candidate: str, reference: str) -> float:
    """
    计算 BLEU 分数（仅中文字符，3-gram，等权重）
    1. 提取候选和参考答案中的中文字符
    2. 将中文字符串视为字符列表（每个汉字为一个 token）
    3. 生成 1-gram、2-gram、3-gram，均匀权重 (1/3, 1/3, 1/3)
    """
    # 提取中文字符
    candidate_zh = extract_chinese_chars(candidate)
    reference_zh = extract_chinese_chars(reference)
    
    if not candidate_zh or not reference_zh:
        return 0.0
    
    # 转换为字符列表
    candidate_tokens = list(candidate_zh)
    reference_tokens = list(reference_zh)
    
    # 生成 n-gram 权重（均匀）
    weights = (1/3, 1/3, 1/3)
    smoothing = SmoothingFunction().method1
    
    # BLEU 要求将 references 包装成列表形式
    references = [reference_tokens]
    
    try:
        bleu = sentence_bleu(
            references,
            candidate_tokens,
            weights=weights,
            smoothing_function=smoothing
        )
        return bleu
    except Exception as e:
        print(f"计算 BLEU 时出错: {e}")
        return 0.0

def check_answer_correctness(question: str, generated_answer: str, reference_answer: str) -> bool:
    prompt = f"""
    你是一个专业的评判员。我会给你一个问题、一个参考答案和一个待评价的答案。
    你的任务是判断待评价答案是否与参考答案一致。你可以容忍一些措辞上的差异。
    但对于参考答案表示信息不足等情况一律判断为"INCORRECT"
    请严格只回复 "CORRECT" 或 "INCORRECT"。

    问题: {question}

    参考标准答案: {reference_answer}

    模型生成答案: {generated_answer}
    """

    try:
        response = eval_client.chat.completions.create(
            model=llm,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        result_text = response.choices[0].message.content.strip().upper()
        return result_text == "CORRECT"
    except Exception as e:
        print(f"调用大模型进行准确性评估时出错: {e}. 问题: '{question[:50]}...'. 将此视为错误。")
        return False

def evaluate_from_test_data():
    try:
        rag_system = RetrieverGenerator()
        
        collection_name = rag_system.select_collection()
        if not collection_name:
            print("无法选择集合，退出评估")
            return
        
        test_files = select_test_datasets()
        if not test_files:
            print("未选择测试数据集，退出评估")
            return
        
        # 创建结果目录
        result_dir = "../data/result"
        os.makedirs(result_dir, exist_ok=True)
        
        # 汇总结果 CSV 文件路径
        csv_path = os.path.join(result_dir, "evaluation_results.csv")
        # 每条查询详细结果 JSONL 文件路径
        jsonl_path = os.path.join(result_dir, "per_query_results.jsonl")
        
        all_mrr = []
        all_ndcg = []
        all_bleu = []
        all_acc = []
        
        # 打开 JSONL 文件（覆盖写入）
        with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
            for file_path in test_files:
                print(f"\n{'='*60}")
                print(f"开始评估数据集: {os.path.basename(file_path)}")
                print(f"{'='*60}")
                
                test_data = load_test_data(file_path)
                if not test_data:
                    continue
                
                mrr_scores = []
                ndcg_scores = []
                bleu_scores = []
                acc_results = []
                
                for i, item in enumerate(tqdm(test_data, desc="处理问题")):
                    query = item["question"]
                    reference_answer = item["answer"]
                    
                    def local_evaluator(generated_ans):
                        return check_answer_correctness(query, generated_ans, reference_answer)
                    
                    generated_answer, final_docs, candidate_docs = rag_system.query(
                        query, 
                        evaluator_func=local_evaluator if ENABLE_ADAPTIVE_RETRIEVAL else None
                    )
                    
                    # 1. ACC
                    is_correct = check_answer_correctness(query, generated_answer, reference_answer)
                    acc_results.append(is_correct)
                    
                    # 2. BLEU
                    bleu_score = calculate_bleu_score(generated_answer, reference_answer)
                    bleu_scores.append(bleu_score)
                    
                    # 3. MRR & NDCG
                    if candidate_docs:
                        all_reranked_for_eval = rag_system.rerank_documents(query, candidate_docs, top_n=len(candidate_docs))
                        mrr = calc_mrr_by_coverage(all_reranked_for_eval, reference_answer, threshold=0.7)
                        ndcg = calc_ndcg_by_coverage(all_reranked_for_eval, reference_answer)
                    else:
                        mrr = 0.0
                        ndcg = 0.0
                    
                    mrr_scores.append(mrr)
                    ndcg_scores.append(ndcg)
                    
                    # 写入 JSONL：每条查询的详细信息
                    per_query_record = {
                        "query": query,
                        "reference_answer": reference_answer,
                        "generated_answer": generated_answer,
                        "acc": 1 if is_correct else 0,
                        "bleu": bleu_score,
                        "mrr": mrr,
                        "ndcg": ndcg
                    }
                    jsonl_file.write(json.dumps(per_query_record, ensure_ascii=False) + '\n')
                    
                    status_str = 'CORRECT' if is_correct else 'INCORRECT'
                    print(f"  问题 #{i+1}: ACC={status_str}, BLEU={bleu_score:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}")
                
                # 数据集汇总
                dataset_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
                dataset_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
                dataset_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
                dataset_acc = sum(acc_results) / len(acc_results) if acc_results else 0
                
                print(f"\n数据集 {os.path.basename(file_path)} 评估结果:")
                print(f"  MRR: {dataset_mrr:.4f}")
                print(f"  NDCG: {dataset_ndcg:.4f}")
                print(f"  BLEU: {dataset_bleu:.4f}")
                print(f"  ACC: {dataset_acc:.4f}")
                
                all_mrr.extend(mrr_scores)
                all_ndcg.extend(ndcg_scores)
                all_bleu.extend(bleu_scores)
                all_acc.extend(acc_results)
        
        # 总体结果
        overall_mrr = sum(all_mrr) / len(all_mrr) if all_mrr else 0
        overall_ndcg = sum(all_ndcg) / len(all_ndcg) if all_ndcg else 0
        overall_bleu = sum(all_bleu) / len(all_bleu) if all_bleu else 0
        overall_acc = sum(all_acc) / len(all_acc) if all_acc else 0
        
        print(f"\n{'='*60}")
        print("总体评估结果:")
        print(f"  MRR: {overall_mrr:.4f}")
        print(f"  NDCG: {overall_ndcg:.4f}")
        print(f"  BLEU: {overall_bleu:.4f}")
        print(f"  ACC: {overall_acc:.4f}")
        print(f"{'='*60}")
        
        # 保存汇总 CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["MRR", f"{overall_mrr:.4f}"])
            writer.writerow(["NDCG", f"{overall_ndcg:.4f}"])
            writer.writerow(["BLEU", f"{overall_bleu:.4f}"])
            writer.writerow(["ACC", f"{overall_acc:.4f}"])
        
        print(f"\n评估结果已保存到:")
        print(f"  汇总指标: {os.path.abspath(csv_path)}")
        print(f"  逐条详情: {os.path.abspath(jsonl_path)}")
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

def interactive_query():
    try:
        rag_system = RetrieverGenerator()
        
        collection_name = rag_system.select_collection()
        if not collection_name:
            print("无法选择集合，退出查询")
            return
        
        print("="*60)
        print("RAG 问答系统")
        print("="*60)
        mode_str = "自适应重试模式" if ENABLE_ADAPTIVE_RETRIEVAL else "标准模式"
        print(f"当前模式: {mode_str}")
        print(f"基础初始召回: {rag_system.initial_retrieve_k} 个片段 | 基础重排序后使用: {rag_system.final_top_k} 个片段")
        if ENABLE_ADAPTIVE_RETRIEVAL:
            try:
                from settings import RETRIEVAL_STEP_SIZE, RERANK_OUTPUT_STEP_SIZE, MAX_RETRIEVAL_ROUNDS
                print(f"重试策略: 每次失败 K+{RETRIEVAL_STEP_SIZE}, TopN+{RERANK_OUTPUT_STEP_SIZE}, 最大轮次 {MAX_RETRIEVAL_ROUNDS}")
            except ImportError:
                print("自适应重试参数未导入，请检查 settings.py")
        print(f"查询日志将保存至: {os.path.abspath(rag_system.log_file_path)}")
        print("输入 'quit' 或 'exit' 退出系统")
        print()
        
        while True:
            try:
                user_input = input("请输入您的问题: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                
                if not user_input:
                    continue
                
                start_time = time.time()
                response, _, _ = rag_system.query(user_input)
                elapsed = time.time() - start_time
                
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
        print("请确保已正确配置 settings.py 中的 llm_API_key")
    except Exception as e:
        print(f"启动系统时出错: {e}")

def main():
    print("="*30)
    print("RAG 系统评估接口")
    print("="*30)
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