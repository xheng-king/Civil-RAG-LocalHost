#!/usr/bin/env python3
"""
生成RAG系统指标对比图：对于4个指标×2个难度，分别为每个系统和每个难度-指标对生成独立的折线图。
文件夹结构：
    image/
        easy_MRR/
            baseline.png
            embedding-ft.png
            adaptive.png
            adaptive_embeddingft.png
        easy_NDCG/
            ...
        hard_MRR/
            ...
        ...
特殊处理：
    - 对于 ACC 指标，纵坐标范围固定为 [0, 1]，且只显示刻度 0 和 1。
    - 横坐标刻度只显示5的倍数序号（步长为5）。
    - 所有字体放大（标题16，轴标签14，刻度13，图例13）。
    - 图片宽度20英寸，高度6英寸。
    - 图片文件若已存在则直接覆盖。
交互式功能：列出当前目录下所有.jsonl文件，让用户为以下8个配置分别选择对应的文件：
    - baseline_easy, baseline_hard
    - embedding-ft_easy, embedding-ft_hard
    - adaptive_easy, adaptive_hard
    - adaptive_embeddingft_easy, adaptive_embeddingft_hard
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ----------------------------- 配置 ---------------------------------
DATA_DIR = Path(".")  # 当前目录，可修改为具体路径
OUTPUT_BASE_DIR = DATA_DIR / "image"  # 图片根目录
SYSTEMS = ["baseline", "embedding-ft", "adaptive", "adaptive_embeddingft"]
DIFFICULTIES = ["easy", "hard"]
METRICS = ["mrr", "ndcg", "bleu", "acc"]
METRIC_NAMES = {"mrr": "MRR", "ndcg": "NDCG", "bleu": "BLEU", "acc": "Accuracy"}

# 颜色方案
COLORS = {"baseline": "blue", "embedding-ft": "green", "adaptive": "orange", "adaptive_embeddingft": "red"}

# 定义8个配置，顺序将决定交互提示的顺序
CONFIG_ITEMS = [
    ("baseline", "easy"),
    ("baseline", "hard"),
    ("embedding-ft", "easy"),
    ("embedding-ft", "hard"),
    ("adaptive", "easy"),
    ("adaptive", "hard"),
    ("adaptive_embeddingft", "easy"),
    ("adaptive_embeddingft", "hard"),
]

def get_all_jsonl_files():
    """返回当前DATA_DIR下所有.jsonl文件的排序列表"""
    files = sorted(DATA_DIR.glob("*.jsonl"))
    return files

def display_files(files):
    """打印文件列表，带序号"""
    print("\n当前目录下的.jsonl文件：")
    for idx, f in enumerate(files, start=1):
        print(f"  {idx}. {f.name}")

def select_file_for_config(config_name, files, selected_map):
    """
    交互式让用户为一个配置选择一个文件。
    config_name: 字符串，如 "naive_easy"
    files: 文件列表
    selected_map: 已选择的映射 {file_index: config_name} 用于检查重复
    返回选中的Path对象
    """
    while True:
        try:
            choice = input(f"\n请为 {config_name} 选择文件序号 (1-{len(files)}): ").strip()
            if not choice:
                print("输入不能为空，请重新输入")
                continue
            idx = int(choice)
            if idx < 1 or idx > len(files):
                print(f"序号超出范围，请输入 1-{len(files)} 之间的数字")
                continue
            if idx in selected_map:
                print(f"警告：文件 {files[idx-1].name} 已经被选为 {selected_map[idx]}，请重新选择（不允许重复）")
                continue
            return files[idx-1], idx
        except ValueError:
            print("请输入有效的数字")

def load_metric_data_from_file(filepath, metric):
    """从指定文件中加载某个指标的值列表，按文件顺序返回"""
    if not filepath.exists():
        print(f"警告：文件 {filepath} 不存在")
        return []
    values = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            val = data.get(metric, None)
            if val is not None:
                values.append(float(val))
    return values

def plot_single_system(metric, difficulty, system, filepath):
    """
    为单个系统、单个难度、单个指标生成一张折线图。
    保存路径: image/{difficulty}_{METRIC_NAMES[metric]}/{system}.png
    特殊处理：
        - 图片宽度20英寸，高度6英寸。
        - 横坐标刻度只显示5的倍数序号。
        - 所有字体放大。
        - 如果 metric == 'acc'，纵坐标范围固定为 [0,1]，只显示刻度 0 和 1。
    """
    values = load_metric_data_from_file(filepath, metric)
    if not values:
        print(f"跳过 {system}_{difficulty} for metric {metric}，无数据")
        return False
    
    # 创建目标文件夹
    folder_name = f"{difficulty}_{METRIC_NAMES[metric]}"
    target_dir = OUTPUT_BASE_DIR / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备横坐标（从 1 到 N）
    x = np.arange(1, len(values) + 1)
    
    # 绘图：增大横向尺寸
    plt.figure(figsize=(20, 6))
    plt.plot(x, values, marker='o', linestyle='-', linewidth=1.5, markersize=4,
             color=COLORS.get(system, "black"), label=system)
    
    # 横坐标：只显示5的倍数序号，字体放大
    tick_positions = np.arange(1, len(values) + 1, 5)
    tick_labels = [str(int(pos)) for pos in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=13, ha='right')
    
    plt.xlabel("Query Index", fontsize=14)
    plt.ylabel(METRIC_NAMES[metric], fontsize=14)
    title = f"{METRIC_NAMES[metric]} - {difficulty.capitalize()} - {system}"
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 特殊处理 ACC：纵坐标范围 [0,1] 且只显示 0 和 1
    if metric == 'acc':
        plt.ylim(0, 1)
        plt.yticks([0, 1], fontsize=13)
    else:
        plt.yticks(fontsize=13)
    
    plt.legend(fontsize=13)
    plt.tight_layout()
    
    out_file = target_dir / f"{system}.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"已生成: {out_file}")
    return True

def main():
    # 1. 获取所有jsonl文件
    jsonl_files = get_all_jsonl_files()
    if len(jsonl_files) == 0:
        print("错误：当前目录下没有找到任何 .jsonl 文件。")
        return
    
    display_files(jsonl_files)
    
    # 2. 交互式选择8个配置对应的文件
    selected_mapping = {}
    selected_index_map = {}
    
    for system, difficulty in CONFIG_ITEMS:
        config_name = f"{system}_{difficulty}"
        selected_file, idx = select_file_for_config(config_name, jsonl_files, selected_index_map)
        selected_mapping[(system, difficulty)] = selected_file
        selected_index_map[idx] = config_name
    
    print("\n文件选择完成！映射如下：")
    for (system, difficulty), path in selected_mapping.items():
        print(f"  {system}_{difficulty} -> {path.name}")
    
    # 3. 为每个指标、每个难度、每个系统生成独立图片
    for difficulty in DIFFICULTIES:
        for metric in METRICS:
            for system in SYSTEMS:
                filepath = selected_mapping.get((system, difficulty))
                if filepath is None:
                    print(f"跳过 {system}_{difficulty} for {metric}，未选择文件")
                    continue
                plot_single_system(metric, difficulty, system, filepath)
    
    print(f"\n所有图表已生成并保存到文件夹: {OUTPUT_BASE_DIR}/")

if __name__ == "__main__":
    main()