#!/usr/bin/env python3
"""
cal_result.py - 计算单个RAG评估结果文件中各指标的均值。
功能：
    1. 列出当前目录下所有 .jsonl 文件，让用户选择一个。
    2. 读取该文件，提取每条数据中的指标（mrr, ndcg, bleu, acc, retrieval_rounds）。
    3. 计算每个指标（包括平均检索次数）的均值并打印到终端，统一保留四位小数。
"""

import json
from pathlib import Path

# 需要计算的指标及其显示名称
METRICS = {
    "mrr": "MRR",
    "ndcg": "NDCG",
    "bleu": "BLEU",
    "acc": "Accuracy",
    "retrieval_rounds": "Avg Retrieval Rounds"
}

def get_jsonl_files():
    """返回当前目录下所有 .jsonl 文件的排序列表"""
    return sorted(Path(".").glob("*.jsonl"))

def display_files(files):
    """打印带序号的可用文件列表"""
    print("\n当前目录下的 .jsonl 文件：")
    for idx, f in enumerate(files, start=1):
        print(f"  {idx}. {f.name}")

def select_file(files):
    """交互式让用户选择一个文件，返回 Path 对象"""
    while True:
        try:
            choice = input(f"\n请选择一个文件（输入序号 1-{len(files)}）: ").strip()
            if not choice:
                print("输入不能为空，请重新输入")
                continue
            idx = int(choice)
            if idx < 1 or idx > len(files):
                print(f"序号超出范围，请输入 1-{len(files)} 之间的数字")
                continue
            return files[idx - 1]
        except ValueError:
            print("请输入有效的数字")

def compute_metrics_mean(filepath):
    """
    读取 JSONL 文件，计算各指标的均值。
    返回字典 {metric_key: mean_value} 和 {metric_key: count}。
    """
    sums = {metric: 0.0 for metric in METRICS}
    counts = {metric: 0 for metric in METRICS}

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告：第 {line_num} 行 JSON 解析失败，已跳过。错误：{e}")
                continue

            for metric in METRICS:
                val = data.get(metric)
                if val is not None:
                    try:
                        num_val = float(val)
                        sums[metric] += num_val
                        counts[metric] += 1
                    except (ValueError, TypeError):
                        print(f"警告：第 {line_num} 行的指标 {metric} 值 '{val}' 无法转换为数字，已跳过")

    # 计算均值，统一保留四位小数（若有效记录数为0则为None）
    means = {}
    for metric in METRICS:
        if counts[metric] > 0:
            mean_val = sums[metric] / counts[metric]
            means[metric] = round(mean_val, 4)
        else:
            means[metric] = None

    return means, counts

def main():
    # 获取文件列表
    jsonl_files = get_jsonl_files()
    if not jsonl_files:
        print("错误：当前目录下没有找到任何 .jsonl 文件。")
        return

    display_files(jsonl_files)
    selected_file = select_file(jsonl_files)
    print(f"\n已选择文件：{selected_file.name}")

    # 计算均值
    means, counts = compute_metrics_mean(selected_file)

    # 打印结果
    print("\n" + "=" * 60)
    print(f"评估指标均值（基于 {selected_file.name}）")
    print("=" * 60)
    for metric, display_name in METRICS.items():
        mean_val = means[metric]
        cnt = counts[metric]
        if mean_val is not None:
            print(f"{display_name:22} : {mean_val:8.4f}  (有效记录数: {cnt})")
        else:
            print(f"{display_name:22} : 无有效数据")
    print("=" * 60)

if __name__ == "__main__":
    main()