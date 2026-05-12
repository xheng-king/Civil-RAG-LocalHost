#!/usr/bin/env python3
"""
生成RAG系统指标对比图：对于4个指标×2个难度，绘制4种系统的单样本指标折线图。
文件命名约定：
    - naive_easy.jsonl, naive_hard.jsonl
    - embedding-ft_easy.jsonl, embedding-ft_hard.jsonl
    - adaptive_easy.jsonl, adaptive_hard.jsonl
    - adaptive_embeddingft_easy.jsonl, adaptive_embeddingft_hard.jsonl
每一行JSON包含字段：mrr, ndcg, bleu, acc
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ----------------------------- 配置 ---------------------------------
DATA_DIR = Path(".")  # 当前目录，也可修改为具体路径
SYSTEMS = ["naive", "embedding-ft", "adaptive", "adaptive_embeddingft"]
DIFFICULTIES = ["easy", "hard"]
METRICS = ["mrr", "ndcg", "bleu", "acc"]
METRIC_NAMES = {"mrr": "MRR", "ndcg": "NDCG", "bleu": "BLEU", "acc": "Accuracy"}

# 颜色方案（可选）
COLORS = {"naive": "blue", "embedding-ft": "green", "adaptive": "orange", "adaptive_embeddingft": "red"}

def load_metric_data(system, difficulty, metric):
    """加载特定系统和难度下的指定指标值列表，按文件顺序返回"""
    filename = f"{system}_{difficulty}.jsonl"
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"警告：文件 {filename} 不存在，跳过")
        return []
    values = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # 注意 json 中的字段可能为 "mrr", "ndcg", "bleu", "acc"
            val = data.get(metric, None)
            if val is not None:
                values.append(float(val))
    return values

def plot_metric_for_difficulty(metric, difficulty):
    """为指定指标和难度生成一张图，包含四种系统的单次指标曲线"""
    plt.figure(figsize=(12, 6))
    for system in SYSTEMS:
        values = load_metric_data(system, difficulty, metric)
        if not values:
            print(f"跳过 {system}_{difficulty}，无数据")
            continue
        x = np.arange(1, len(values) + 1)
        plt.plot(x, values, marker='o', linestyle='-', linewidth=1.5, markersize=3,
                 label=system, color=COLORS.get(system))
    plt.xlabel("Query Index")
    plt.ylabel(METRIC_NAMES[metric])
    title = f"{METRIC_NAMES[metric]} - {difficulty.capitalize()} Set"
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    # 保存图片
    out_filename = f"{metric}_{difficulty}.png"
    plt.tight_layout()
    plt.savefig(out_filename, dpi=150)
    plt.close()
    print(f"已生成: {out_filename}")

def main():
    # 确保当前目录或指定目录存在
    if not DATA_DIR.exists():
        print(f"错误：目录 {DATA_DIR} 不存在")
        return
    for metric in METRICS:
        for difficulty in DIFFICULTIES:
            plot_metric_for_difficulty(metric, difficulty)
    print("所有图表生成完毕！")

if __name__ == "__main__":
    main()