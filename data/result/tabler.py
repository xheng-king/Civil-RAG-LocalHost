#!/usr/bin/env python3
"""
将RAG系统评估结果汇总到一个Excel文件中。
每个Sheet的命名格式：<难度>_<指标>，例如 easy_MRR, hard_MRR, easy_NDCG, hard_NDCG等。
每个Sheet包含列：Query, Reference Answer, 以及四个系统的指标值。
BLEU和NDCG数值保留两位小数。

交互式功能：列出当前目录下所有.jsonl文件，让用户为以下8个配置分别选择对应的文件：
- naive_easy
- naive_hard
- embedding-ft_easy
- embedding-ft_hard
- adaptive_easy
- adaptive_hard
- adaptive_embeddingft_easy
- adaptive_embeddingft_hard
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

# 定义8个配置，顺序将决定交互提示的顺序
CONFIG_ITEMS = [
    ("naive", "easy"),
    ("naive", "hard"),
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
            # 检查是否已被其他配置选择
            if idx in selected_map:
                print(f"警告：文件 {files[idx-1].name} 已经被选为 {selected_map[idx]}，请重新选择（不允许重复）")
                continue
            return files[idx-1], idx
        except ValueError:
            print("请输入有效的数字")

def main():
    # 1. 获取所有jsonl文件
    jsonl_files = get_all_jsonl_files()
    if len(jsonl_files) == 0:
        print("错误：当前目录下没有找到任何 .jsonl 文件。")
        return
    
    display_files(jsonl_files)
    
    # 2. 交互式选择8个配置对应的文件
    selected_mapping = {}  # key: (system, difficulty), value: Path
    selected_index_map = {}  # key: file_index, value: config_name (用于重复检查)
    
    for system, difficulty in CONFIG_ITEMS:
        config_name = f"{system}_{difficulty}"
        selected_file, idx = select_file_for_config(config_name, jsonl_files, selected_index_map)
        selected_mapping[(system, difficulty)] = selected_file
        selected_index_map[idx] = config_name
    
    print("\n文件选择完成！映射如下：")
    for (system, difficulty), path in selected_mapping.items():
        print(f"  {system}_{difficulty} -> {path.name}")
    
    # 3. 数据加载函数（使用用户选择的文件路径）
    def load_system_data(system, difficulty, metric):
        """
        读取指定系统和难度对应的文件，返回三个列表：
        - queries: 查询文本列表
        - references: 参考答案列表
        - values: 对应指标值列表（可能为None若缺失）
        """
        filepath = selected_mapping.get((system, difficulty))
        if filepath is None or not filepath.exists():
            print(f"警告：文件 {filepath} 不存在或未选择，跳过")
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
            # 注意：某些系统可能没有对应的文件（理论上用户已选，但以防万一）
            if (system, difficulty) not in selected_mapping:
                continue
            queries, refs, values = load_system_data(system, difficulty, metric)
            if not queries:  # 如果文件为空，跳过该系统
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
                # 如果长度不足max_len，后面补None
                values_padded = values + [None] * (max_len - len(values))
                df_dict[f"{system}"] = values_padded
            else:
                # 系统文件未选择或不存在，填充None
                df_dict[f"{system}"] = [None] * max_len
        
        df = pd.DataFrame(df_dict)
        return df
    
    # 4. 生成Excel
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        for difficulty in DIFFICULTIES:
            for metric in METRICS:
                sheet_name = f"{difficulty}_{METRIC_DISPLAY[metric]}"
                df = build_dataframe_for_metric_difficulty(metric, difficulty)
                if df is not None and not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"已写入: {sheet_name}")
                else:
                    print(f"跳过: {sheet_name} (无数据)")
    print(f"\nExcel文件已生成: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()