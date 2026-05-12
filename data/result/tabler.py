#!/usr/bin/env python3
"""
将RAG系统评估结果汇总到一个Excel文件中。
每个Sheet的命名格式：<难度>_<指标>，例如 easy_MRR, hard_MRR, easy_NDCG, hard_NDCG等。
每个Sheet包含列：Query, Reference Answer, 以及四个系统的指标值。
BLEU和NDCG数值保留两位小数。
"""

import json
import pandas as pd
from pathlib import Path

# ================= 配置 ==================
DATA_DIR = Path(".")  # 当前目录，可修改为具体路径
OUTPUT_FILE = "metrics_summary.xlsx"

SYSTEMS = ["naive", "embedding-ft", "adaptive", "adaptive_embeddingft"]
DIFFICULTIES = ["easy", "hard"]
METRICS = ["mrr", "ndcg", "bleu", "acc"]
METRIC_DISPLAY = {"mrr": "MRR", "ndcg": "NDCG", "bleu": "BLEU", "acc": "Accuracy"}

# 需要保留两位小数的指标
DECIMAL_METRICS = ["ndcg", "bleu"]

def load_system_data(system, difficulty, metric):
    """
    读取指定系统和难度下的JSONL文件，返回四个列表：
    - queries: 查询文本列表
    - references: 参考答案列表
    - values: 对应指标值列表（可能为None若缺失）
    """
    filename = f"{system}_{difficulty}.jsonl"
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"警告：文件 {filename} 不存在，跳过")
        return [], [], []
    
    queries = []
    references = []
    values = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            queries.append(data.get("query", ""))
            references.append(data.get("reference_answer", ""))
            val = data.get(metric, None)
            # 对于BLEU和NDCG，格式化为两位小数（如果非None）
            if val is not None and metric in DECIMAL_METRICS:
                val = round(float(val), 2)
            values.append(val)
    return queries, references, values

def build_dataframe_for_metric_difficulty(metric, difficulty):
    """
    为一个指标+难度构建DataFrame，
    行索引按所有系统中该难度文件的最大长度对齐，若某系统缺失行则填充None。
    """
    # 收集所有系统的数据和长度
    system_data = {}
    max_len = 0
    for system in SYSTEMS:
        queries, refs, values = load_system_data(system, difficulty, metric)
        if not queries:  # 如果文件不存在或为空，跳过该系统
            continue
        system_data[system] = (queries, refs, values)
        max_len = max(max_len, len(queries))
    
    if not system_data:
        print(f"警告：难度 {difficulty} 指标 {metric} 无任何有效数据，跳过")
        return None
    
    # 检查所有文件的query/reference是否一致（按行比较）
    base_queries = None
    base_refs = None
    for system, (queries, refs, _) in system_data.items():
        if base_queries is None:
            base_queries = queries
            base_refs = refs
        else:
            if queries != base_queries:
                print(f"警告：{system}_{difficulty} 与首个文件的query列表不一致，将按行索引对齐（可能错位）")
            if refs != base_refs:
                print(f"警告：{system}_{difficulty} 与首个文件的reference列表不一致，将按行索引对齐")
    
    # 构建DataFrame的基础列
    # 取第一个系统的queries和refs作为基准，如果所有系统都没有数据，则跳过
    if base_queries is None:
        return None
    
    df_dict = {
        "Query": base_queries,
        "Reference Answer": base_refs
    }
    # 添加每个系统的指标列
    for system in SYSTEMS:
        if system in system_data:
            _, _, values = system_data[system]
            # 如果长度不足max_len，后面补None（通常不会发生，但安全）
            values_padded = values + [None] * (max_len - len(values))
            df_dict[f"{system}"] = values_padded
        else:
            # 系统文件不存在，填充None
            df_dict[f"{system}"] = [None] * max_len
    
    df = pd.DataFrame(df_dict)
    # 按照原始顺序（文件顺序）保留行，不需要额外排序
    return df

def main():
    # 创建Excel写入器
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for difficulty in DIFFICULTIES:
            for metric in METRICS:
                sheet_name = f"{difficulty}_{METRIC_DISPLAY[metric]}"
                # Excel sheet名称长度限制31字符，我们的名称通常很短，安全
                df = build_dataframe_for_metric_difficulty(metric, difficulty)
                if df is not None and not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"已写入: {sheet_name}")
                else:
                    print(f"跳过: {sheet_name} (无数据)")
    print(f"Excel文件已生成: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()