# auto_label.py

"""
自动使用 rerank 模型为训练数据生成标签。
对于每个 query 及其关联的多个 response（1 个原始正例 + BASE_INITIAL_RETRIEVE_K 个检索负例），
调用 rerank 模型打分，取 top-2 的 response 标记为 1，其余为 0。
输出文件保存在 data/training/ 下，文件名为 {原文件名_stem}_labeled.jsonl。
支持断点续标：若输出文件已存在，跳过其中已标注过的 query。
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import requests
from typing import List, Tuple, Dict, Any, Set

# 添加项目根目录到 sys.path，以便导入 settings
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入配置
try:
    from src.settings import (
        rerank_model,
        rerank_base_url,
        rerank_API_key,
        BASE_INITIAL_RETRIEVE_K
    )
except ImportError:
    print("错误：无法从 src.settings 导入必要的配置。")
    sys.exit(1)

def list_training_jsonl_files() -> List[Path]:
    """列出 data/training 目录下的所有 .jsonl 文件"""
    training_dir = project_root / "data" / "training"
    if not training_dir.exists():
        print(f"目录不存在: {training_dir}")
        return []
    files = list(training_dir.glob("*.jsonl"))
    return files

def select_file(files: List[Path]) -> Path:
    """交互式选择文件"""
    if not files:
        print("没有找到任何 .jsonl 文件。")
        sys.exit(0)
    print("可用的训练数据文件（data/training/*.jsonl）：")
    for idx, f in enumerate(files, 1):
        print(f"{idx}. {f.name}")
    while True:
        try:
            choice = int(input("请选择文件编号: "))
            if 1 <= choice <= len(files):
                return files[choice-1]
            else:
                print(f"请输入 1 ~ {len(files)} 之间的数字。")
        except ValueError:
            print("输入无效，请输入数字。")

def load_original_data(filepath: Path) -> Dict[str, List[Tuple[str, float]]]:
    """
    读取原始 jsonl 文件，按 query 分组，返回每个 query 对应的 response 列表（保持原顺序）。
    原始格式每行: {"query": "...", "response": "...", "label": ...}
    label 字段将被忽略，但会保留 response 原始顺序。
    """
    groups = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                query = data.get('query')
                response = data.get('response')
                if query is None or response is None:
                    print(f"警告：第 {line_num} 行缺少 query 或 response 字段，跳过。")
                    continue
                groups[query].append((response, data.get('label', 0.0)))
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_num} 行 JSON 解析失败: {e}")
    return groups

def load_labeled_queries(output_path: Path) -> Set[str]:
    """
    读取已有的标注输出文件，返回其中所有出现过的 query 集合。
    只要 query 出现过一次，就认为该 query 已被标注，后续跳过。
    """
    labeled_queries = set()
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    q = data.get('query')
                    if q:
                        labeled_queries.add(q)
                except:
                    pass
    return labeled_queries

def call_rerank(query: str, documents: List[str]) -> List[float]:
    """
    调用 rerank 模型，返回每个文档与查询的相关性分数（浮点数，越大越相关）。
    基于项目现有的 rerank_documents 方法实现，适配 settings 中的配置。
    """
    docs_dict = [{"content": doc, "initial_distance": 0.0} for doc in documents]

    headers = {
        "Authorization": f"Bearer {rerank_API_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": rerank_model,
        "documents": [d['content'] for d in docs_dict],
        "query": query,
        "top_n": len(documents)
    }

    try:
        response = requests.post(
            rerank_base_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if 'results' not in result:
            print(f"API 响应缺少 'results' 字段: {result}")
            return [0.0] * len(documents)

        api_results = result['results']
        scores = [0.0] * len(documents)
        for rank_data in api_results:
            idx = rank_data.get('index')
            score = rank_data.get('relevance_score')
            if idx is not None and score is not None:
                scores[idx] = float(score)
        return scores

    except Exception as e:
        print(f"调用 rerank API 失败: {e}")
        return [0.0] * len(documents)

def process_and_write_one_query(
    query: str,
    resp_list: List[Tuple[str, float]],
    output_path: Path,
    expected_responses_per_query: int,
    idx: int,
    total: int
) -> bool:
    """
    处理单个 query：
    - 调用 rerank 获取分数
    - 确定每个 response 的标签（top-2 为 1，其余 0）
    - 立即追加写入输出文件
    返回 True 表示成功，False 表示跳过（如响应数量异常且用户选择跳过）
    """
    responses = [r[0] if isinstance(r, tuple) else r for r in resp_list]
    if len(responses) != expected_responses_per_query:
        print(f"警告：Query '{query[:50]}...' 有 {len(responses)} 个 response，"
              f"预期 {expected_responses_per_query} 个（1 + BASE_INITIAL_RETRIEVE_K）。")
        # 可选择是否继续处理，这里默认继续（但会提示）
        user_choice = input("是否继续处理该 query？(y/N): ").strip().lower()
        if user_choice != 'y':
            print("跳过此 query。")
            return False

    print(f"处理 {idx}/{total}: {query[:60]}... (共 {len(responses)} 个候选)")

    scores = call_rerank(query, responses)
    if len(scores) == 0:
        print("未获取到有效分数，跳过。")
        return False

    # 取分数最高的前两个索引
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top2_indices = set(sorted_indices[:2])

    # 生成当前 query 的所有三元组
    entries = []
    for i, resp in enumerate(responses):
        label = 1 if i in top2_indices else 0
        entries.append({
            "query": query,
            "response": resp,
            "label": label
        })

    # 追加写入输出文件（一次性写入该 query 的所有行）
    with open(output_path, 'a', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        f.flush()
    print(f"已写入 {len(entries)} 条记录（query 完成）")
    return True

def main():
    # 列出现有文件
    files = list_training_jsonl_files()
    if not files:
        print("data/training 目录下没有找到 .jsonl 文件。")
        sys.exit(0)

    selected = select_file(files)
    print(f"已选择: {selected.name}")

    # 读取原始数据，按 query 分组
    all_groups = load_original_data(selected)
    if not all_groups:
        print("文件中没有有效的 query-response 对。")
        sys.exit(0)

    # 确定输出文件路径
    output_path = selected.parent / f"{selected.stem}_labeled.jsonl"

    # 加载已标注的 query 集合（断点续标）
    labeled_queries = load_labeled_queries(output_path)
    if labeled_queries:
        print(f"输出文件已存在，其中包含 {len(labeled_queries)} 个已标注的 query，将跳过它们。")

    # 过滤出未标注的 query
    pending_groups = {q: resp_list for q, resp_list in all_groups.items() if q not in labeled_queries}
    if not pending_groups:
        print("所有 query 均已标注，无需处理。")
        sys.exit(0)

    expected_k = BASE_INITIAL_RETRIEVE_K + 1
    print(f"期望每个 query 有 {expected_k} 个 response（1 个原始 + {BASE_INITIAL_RETRIEVE_K} 个检索负例）")
    print(f"本次需要处理 {len(pending_groups)} 个新 query。")

    total = len(pending_groups)
    for idx, (query, resp_list) in enumerate(pending_groups.items(), 1):
        process_and_write_one_query(
            query, resp_list, output_path, expected_k, idx, total
        )
        # 可选：每处理完一个 query 后短暂延时，避免 API 限流
        # time.sleep(0.5)

    print(f"全部处理完成！结果已追加保存至: {output_path}")

if __name__ == "__main__":
    main()