# rag_interface.py
import os
import chromadb
from openai import OpenAI
from typing import List, Dict, Any
from database_manager import DatabaseManager
from retriever_generator import QwenRetrieverGenerator
import time
import json
from datetime import datetime
import csv
import glob
import math
from collections import Counter
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

def calculate_dcg_from_scores(scores):
    """根据分数列表计算 DCG，分数直接作为增益"""
    dcg = 0.0
    for i, score in enumerate(scores):
        gain = 2 ** score - 1
        dcg += gain / math.log2(i + 2)
    return dcg

def calc_mrr(scores):
    """根据分数列表计算 MRR（最高分文档视为唯一相关文档）"""
    if not scores:
        return 0.1
    # 分数越高越好，找到最高分的索引（0‑based），排名为 index+1
    max_idx = max(range(len(scores)), key=lambda i: scores[i])
    rank = max_idx + 1
    return 1.0 / rank if rank <= 5 else 0.1

def calc_ndcg(scores):
    """根据分数列表计算 NDCG"""
    if not scores:
        return 0.0
    dcg = calculate_dcg_from_scores(scores)
    # 理想排序：将分数降序排列
    ideal_scores = sorted(scores, reverse=True)
    idcg = calculate_dcg_from_scores(ideal_scores)
    return dcg / idcg if idcg > 0 else 0.0

def calculate_bleu_score(candidate, reference, max_n=4):
    if not candidate or not reference:
        return 0.0

    try:
        candidate_tokens = jieba.lcut(candidate.strip())
        reference_tokens = jieba.lcut(reference.strip())

        if not isinstance(candidate_tokens, list) or not isinstance(reference_tokens, list):
            print(f"分词结果类型错误: candidate type={type(candidate_tokens)}, reference type={type(reference_tokens)}")
            return 0.0

        if len(candidate_tokens) == 0:
            return 0.0

        weights = (0.25, 0.25, 0.25, 0.25)
        chencherry = SmoothingFunction()
        
        references_for_nltk = [reference_tokens]
        hypothesis_for_nltk = candidate_tokens
        
        bleu_score = sentence_bleu(
            references=references_for_nltk,
            hypothesis=hypothesis_for_nltk,
            weights=weights,
            smoothing_function=chencherry.method1
        )

        return bleu_score

    except Exception as e:
        import traceback
        print(f"计算 BLEU 时发生异常: {e}")
        traceback.print_exc()
        print(f"  Candidate: '{candidate}', Reference: '{reference}'")
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
        if "CORRECT" == result_text:
            return True
        else:
            return False
    except Exception as e:
        print(f"调用大模型进行准确性评估时出错: {e}. 问题: '{question[:50]}...'. 将此视为错误。")
        return False

def evaluate_from_test_data():
    try:
        rag_system = QwenRetrieverGenerator()
        
        collection_name = rag_system.select_collection()
        if not collection_name:
            print("无法选择集合，退出评估")
            return
        
        test_files = select_test_datasets()
        if not test_files:
            print("未选择测试数据集，退出评估")
            return
        
        all_mrr = []
        all_ndcg = []
        all_bleu = []
        all_acc = []
        
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
                
                # 定义一个闭包评估器，捕获当前的 reference_answer
                def local_evaluator(generated_ans):
                    return check_answer_correctness(query, generated_ans, reference_answer)
                
                # 调用 query，获取答案以及最终轮的文档信息
                # generated_answer, final_docs (for LLM), candidate_docs (before rerank)
                generated_answer, final_docs, candidate_docs = rag_system.query(query, evaluator_func=local_evaluator if ENABLE_ADAPTIVE_RETRIEVAL else None)

                # 1. ACC
                is_correct = check_answer_correctness(query, generated_answer, reference_answer)
                acc_results.append(is_correct)
                
                # 2. BLEU
                bleu_score = calculate_bleu_score(generated_answer, reference_answer)
                bleu_scores.append(bleu_score)
                
                # 3. MRR & NDCG
                # 基于最终那一轮（成功或最后一轮）的 candidate_docs 和它们的 rerank_score
                if candidate_docs:
                                        pass 

                
                if candidate_docs:
                    # 重新获取所有文档的重排序分数
                    all_reranked_for_eval = rag_system.rerank_documents(query, candidate_docs, top_n=len(candidate_docs))
                    
                    # 构建分数列表，保持与 candidate_docs 相同的顺序（通过 id 映射）
                    rerank_score_map = {}
                    for rd in all_reranked_for_eval:
                        if 'rerank_score' in rd and rd['rerank_score'] is not None:
                            rerank_score_map[rd['id']] = rd['rerank_score']
                    
                    scores_for_eval = []
                    for doc in candidate_docs:
                        scores_for_eval.append(rerank_score_map.get(doc['id'], 0.0))
                        
                    mrr = calc_mrr(scores_for_eval)
                    ndcg = calc_ndcg(scores_for_eval)
                else:
                    mrr = 0.1
                    ndcg = 0.0

                mrr_scores.append(mrr)
                ndcg_scores.append(ndcg)
                
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
        
        result_file = "../evaluation_results.csv"
        with open(result_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["MRR", f"{overall_mrr:.4f}"])
            writer.writerow(["NDCG", f"{overall_ndcg:.4f}"])
            writer.writerow(["BLEU", f"{overall_bleu:.4f}"])
            writer.writerow(["ACC", f"{overall_acc:.4f}"])
        
        print(f"\n评估结果已保存到: {os.path.abspath(result_file)}")
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()


def interactive_query():
    try:
        rag_system = QwenRetrieverGenerator()
        
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
            print(f"重试策略: 每次失败 K+{RETRIEVAL_STEP_SIZE}, TopN+{RERANK_OUTPUT_STEP_SIZE}, 最大轮次 {MAX_RETRIEVAL_ROUNDS}")
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
                # 交互式查询不传评估器，因此即使启用自适应，也只会执行标准的一轮（根据 retriever_generator 的逻辑）
                # 如果你希望交互式也自适应，你需要提供一个无监督的评估器（例如基于置信度）
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