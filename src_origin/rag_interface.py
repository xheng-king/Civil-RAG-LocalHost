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

from settings import llm, llm_base_url, llm_API_key

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

def calculate_dcg(grades):
    dcg = 0
    for i, grade in enumerate(grades):
        dcg += (2**grade - 1) / math.log2(i + 2)
    return dcg

def calculate_idcg(grades):
    sorted_grades = sorted(grades, reverse=True)
    return calculate_dcg(sorted_grades)

def get_ngrams(text, n):
    tokens = text.split()
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams

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
    你是一个专业的评判员。我会给你一个问题、一个参考标准答案和一个模型生成的答案。
    你的任务是判断模型生成的答案是否正确回答了问题。你可以容忍一些措辞上的差异，但核心意思必须一致。
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
        if "CORRECT" in result_text:
            return True
        elif "INCORRECT" in result_text:
            return False
        else:
            print(f"警告: 大模型评估输出格式不符合预期: '{result_text}'. 问题: '{question[:50]}...'. 将此视为错误。")
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
        
        all_mrr_initial = []
        all_ndcg_initial = []
        all_mrr_reranked = []
        all_ndcg_reranked = []
        all_bleu = []
        all_acc = []
        
        for file_path in test_files:
            print(f"\n{'='*60}")
            print(f"开始评估数据集: {os.path.basename(file_path)}")
            print(f"{'='*60}")
            
            test_data = load_test_data(file_path)
            if not test_data:
                continue
            
            mrr_initial_scores = []
            ndcg_initial_scores = []
            mrr_reranked_scores = []
            ndcg_reranked_scores = []
            bleu_scores = []
            acc_results = []
            
            for i, item in enumerate(tqdm(test_data, desc="处理问题")):
                query = item["question"]
                reference_answer = item["answer"]
                
                candidate_docs = rag_system.retrieve_documents(query)
                
                if not candidate_docs:
                    print(f"  问题 #{i+1}: 未检索到任何文档")
                    mrr_initial_scores.append(0.1)
                    ndcg_initial_scores.append(0)
                    mrr_reranked_scores.append(0.1)
                    ndcg_reranked_scores.append(0)
                    bleu_scores.append(0)
                    acc_results.append(False)
                    continue
                
                first_relevant_rank_initial = rag_system.get_relevance_rank(query, candidate_docs)
                mrr_initial = 1.0 / first_relevant_rank_initial if first_relevant_rank_initial <= 5 else 0.1
                mrr_initial_scores.append(mrr_initial)
                
                grades_initial = []
                for doc in candidate_docs:
                    grade = rag_system.get_relevance_grade(query, doc)
                    grades_initial.append(grade)
                
                dcg_initial = calculate_dcg(grades_initial)
                idcg_initial = calculate_idcg(grades_initial)
                
                ndcg_initial = dcg_initial / idcg_initial if idcg_initial > 0 else 0
                ndcg_initial_scores.append(ndcg_initial)
                
                reranked_docs = rag_system.rerank_documents(query, candidate_docs)
                
                first_relevant_rank_reranked = rag_system.get_relevance_rank(query, reranked_docs)
                mrr_reranked = 1.0 / first_relevant_rank_reranked if first_relevant_rank_reranked <= 5 else 0.1
                mrr_reranked_scores.append(mrr_reranked)
                
                grades_reranked = []
                for doc in reranked_docs:
                    grade = rag_system.get_relevance_grade(query, doc)
                    grades_reranked.append(grade)
                
                dcg_reranked = calculate_dcg(grades_reranked)
                idcg_reranked = calculate_idcg(grades_reranked)
                
                ndcg_reranked = dcg_reranked / idcg_reranked if idcg_reranked > 0 else 0
                ndcg_reranked_scores.append(ndcg_reranked)
                
                final_docs_for_generation = reranked_docs[:rag_system.final_top_k]
                generated_answer = rag_system.generate_answer(query, final_docs_for_generation)

                bleu_score = calculate_bleu_score(generated_answer, reference_answer)
                bleu_scores.append(bleu_score)
                
                is_correct = check_answer_correctness(query, generated_answer, reference_answer)
                acc_results.append(is_correct)
                
                print(f"  问题 #{i+1}: MRR_Init={mrr_initial:.4f}, NDCG_Init={ndcg_initial:.4f}, MRR_Rerank={mrr_reranked:.4f}, NDCG_Rerank={ndcg_reranked:.4f}, BLEU={bleu_score:.4f}, ACC={'CORRECT' if is_correct else 'INCORRECT'}")
            
            dataset_mrr_initial = sum(mrr_initial_scores) / len(mrr_initial_scores) if mrr_initial_scores else 0
            dataset_ndcg_initial = sum(ndcg_initial_scores) / len(ndcg_initial_scores) if ndcg_initial_scores else 0
            dataset_mrr_reranked = sum(mrr_reranked_scores) / len(mrr_reranked_scores) if mrr_reranked_scores else 0
            dataset_ndcg_reranked = sum(ndcg_reranked_scores) / len(ndcg_reranked_scores) if ndcg_reranked_scores else 0
            dataset_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
            dataset_acc = sum(acc_results) / len(acc_results) if acc_results else 0
            
            print(f"\n数据集 {os.path.basename(file_path)} 评估结果:")
            print(f"  MRR_Initial: {dataset_mrr_initial:.4f}")
            print(f"  NDCG_Initial: {dataset_ndcg_initial:.4f}")
            print(f"  MRR_Reranked: {dataset_mrr_reranked:.4f}")
            print(f"  NDCG_Reranked: {dataset_ndcg_reranked:.4f}")
            print(f"  BLEU: {dataset_bleu:.4f}")
            print(f"  ACC: {dataset_acc:.4f}")
            
            all_mrr_initial.extend(mrr_initial_scores)
            all_ndcg_initial.extend(ndcg_initial_scores)
            all_mrr_reranked.extend(mrr_reranked_scores)
            all_ndcg_reranked.extend(ndcg_reranked_scores)
            all_bleu.extend(bleu_scores)
            all_acc.extend(acc_results)
        
        overall_mrr_initial = sum(all_mrr_initial) / len(all_mrr_initial) if all_mrr_initial else 0
        overall_ndcg_initial = sum(all_ndcg_initial) / len(all_ndcg_initial) if all_ndcg_initial else 0
        overall_mrr_reranked = sum(all_mrr_reranked) / len(all_mrr_reranked) if all_mrr_reranked else 0
        overall_ndcg_reranked = sum(all_ndcg_reranked) / len(all_ndcg_reranked) if all_ndcg_reranked else 0
        overall_bleu = sum(all_bleu) / len(all_bleu) if all_bleu else 0
        overall_acc = sum(all_acc) / len(all_acc) if all_acc else 0
        
        print(f"\n{'='*60}")
        print("总体评估结果:")
        print(f"  MRR_Initial: {overall_mrr_initial:.4f}")
        print(f"  NDCG_Initial: {overall_ndcg_initial:.4f}")
        print(f"  MRR_Reranked: {overall_mrr_reranked:.4f}")
        print(f"  NDCG_Reranked: {overall_ndcg_reranked:.4f}")
        print(f"  BLEU: {overall_bleu:.4f}")
        print(f"  ACC: {overall_acc:.4f}")
        print(f"{'='*60}")
        
        result_file = "../evaluation_results.csv"
        with open(result_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["MRR_Initial", f"{overall_mrr_initial:.4f}"])
            writer.writerow(["NDCG_Initial", f"{overall_ndcg_initial:.4f}"])
            writer.writerow(["MRR_Reranked", f"{overall_mrr_reranked:.4f}"])
            writer.writerow(["NDCG_Reranked", f"{overall_ndcg_reranked:.4f}"])
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
        print(f"初始召回: {rag_system.initial_retrieve_k} 个片段 | 重排序后使用: {rag_system.final_top_k} 个片段")
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
                response = rag_system.query(user_input)
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