#!/usr/bin/env python3
"""
compare_rag_metrics.py - 比较两个 RAG 评估结果文件中指定指标的差异。
功能：
    1. 列出当前目录下所有 .jsonl 文件，让用户依次选择两个文件。
    2. 让用户选择要比较的指标：MRR、NDCG、BLEU、Accuracy 或 Avg Retrieval Rounds。
    3. 逐行对齐两个文件（以行数较少者为准），比较每行中该指标的值。
    4. 打印所有差异行的序号，以及两个文件中的具体值（缺失或无效时会标记）。
"""

import json
from pathlib import Path

# 指标映射：显示名称 -> 数据中的 key
METRICS = {
    "1": {"name": "MRR", "key": "mrr"},
    "2": {"name": "NDCG", "key": "ndcg"},
    "3": {"name": "BLEU", "key": "bleu"},
    "4": {"name": "Accuracy", "key": "acc"},
    "5": {"name": "Avg Retrieval Rounds", "key": "retrieval_rounds"},
}

def get_jsonl_files():
    """返回当前目录下所有 .jsonl 文件的排序列表"""
    return sorted(Path(".").glob("*.jsonl"))

def display_files(files):
    """打印带序号的可用文件列表"""
    print("\n当前目录下的 .jsonl 文件：")
    for idx, f in enumerate(files, start=1):
        print(f"  {idx}. {f.name}")

def select_file(files, prompt="请选择一个文件"):
    """交互式让用户选择一个文件，返回 Path 对象"""
    while True:
        try:
            choice = input(f"\n{prompt}（输入序号 1-{len(files)}）: ").strip()
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

def select_metric():
    """让用户选择要比较的指标，返回指标 key"""
    print("\n请选择要比较的指标：")
    for opt, info in METRICS.items():
        print(f"  {opt}. {info['name']}")
    while True:
        choice = input("\n请输入选项数字: ").strip()
        if choice in METRICS:
            return METRICS[choice]["key"]
        print("无效选项，请重新输入")

def load_lines(filepath):
    """
    加载 JSONL 文件，返回列表，每个元素为 (行号, 原始数据字典)，
    行号从 1 开始，解析失败的行会被跳过并给出警告。
    """
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                lines.append((line_num, data))
            except json.JSONDecodeError as e:
                print(f"警告：文件 {filepath.name} 第 {line_num} 行 JSON 解析失败，已跳过。错误：{e}")
                continue
    return lines

def get_metric_value(data, metric_key, line_num, file_name):
    """
    从数据字典中提取指标值（转为 float），若不存在或无法转换则返回 None。
    同时打印警告信息。
    """
    val = data.get(metric_key)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        print(f"警告：文件 {file_name} 第 {line_num} 行的指标 {metric_key} 值 '{val}' 无法转换为数字")
        return None

def compare_files(file1, file2, metric_key):
    """比较两个文件的指定指标，返回差异列表"""
    # 加载两个文件的有效行（跳过解析错误的行）
    lines1 = load_lines(file1)
    lines2 = load_lines(file2)

    # 为了便于按顺序对齐，我们将行号信息保留，但比较时按有效行在文件中的顺序（即排除了错误行）
    # 注意：原文件中跳过的行不会被计入比较，这可能导致用户期望的序号与原始行号不一致。
    # 这里我们采用原始行号进行对齐：如果某一行在某个文件中被跳过，那么该行视为缺失，标记为差异。
    # 更符合直觉的做法：以两个文件的最大行号为基准，但若某行在其中一文件被跳过，则缺失标记。
    # 实现方法：获取两个文件中每个原始行号的数据（如果该行被跳过则无数据）。
    # 构建字典：行号 -> (data, 文件对象)
    # 但由于 load_lines 已经跳过了错误行，我们无法知道被跳过的行号。因此我们需要在读取时记录所有行（包括解析失败的行），
    # 对解析失败的行也记录一个占位符，这样才能对齐原始行号。
    # 重新设计：读取时记录所有行（保留原始行号），对解析失败的行 data=None。

    def load_all_lines(filepath):
        """读取所有行，返回字典 {行号: data_or_None}，行号从1开始"""
        lines_dict = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    lines_dict[line_num] = None
                    continue
                try:
                    data = json.loads(line)
                    lines_dict[line_num] = data
                except json.JSONDecodeError as e:
                    print(f"警告：文件 {filepath.name} 第 {line_num} 行 JSON 解析失败，已跳过。错误：{e}")
                    lines_dict[line_num] = None
        return lines_dict

    dict1 = load_all_lines(file1)
    dict2 = load_all_lines(file2)

    # 取两个文件的最大行号，以便输出所有存在差异的行（包括某一方无对应行的）
    max_line = max(max(dict1.keys(), default=0), max(dict2.keys(), default=0))
    if max_line == 0:
        print("两个文件均无有效数据行，无法比较。")
        return []

    # 比较每一行
    differences = []
    for line_num in range(1, max_line + 1):
        data1 = dict1.get(line_num)
        data2 = dict2.get(line_num)

        # 如果该行在两个文件中都不存在（比如文件行数不够），跳过
        if data1 is None and data2 is None:
            continue

        # 获取指标值
        val1 = None
        if data1 is not None:
            val1 = get_metric_value(data1, metric_key, line_num, file1.name)
        val2 = None
        if data2 is not None:
            val2 = get_metric_value(data2, metric_key, line_num, file2.name)

        # 判断是否不同：值不同，或者其中一个缺失
        if val1 != val2:
            differences.append((line_num, val1, val2))

    return differences

def main():
    # 获取文件列表
    jsonl_files = get_jsonl_files()
    if not jsonl_files:
        print("错误：当前目录下没有找到任何 .jsonl 文件。")
        return

    display_files(jsonl_files)

    # 选择第一个文件
    file1 = select_file(jsonl_files, "请选择第一个文件")
    # 选择第二个文件（不能与第一个相同）
    while True:
        file2 = select_file(jsonl_files, "请选择第二个文件")
        if file2 == file1:
            print("您选择了同一个文件，请重新选择另一个文件。")
        else:
            break

    print(f"\n第一个文件：{file1.name}")
    print(f"第二个文件：{file2.name}")

    # 选择要比较的指标
    metric_key = select_metric()
    # 获取指标显示名称
    metric_display = next(info["name"] for info in METRICS.values() if info["key"] == metric_key)

    print(f"\n正在比较指标：{metric_display} ...")

    # 执行比较
    differences = compare_files(file1, file2, metric_key)

    # 输出结果
    print("\n" + "=" * 70)
    print(f"比较结果（指标：{metric_display}）")
    print("=" * 70)
    if not differences:
        print("两个文件在所有有效行中该指标值完全一致。")
    else:
        print(f"共发现 {len(differences)} 处差异：\n")
        print(f"{'行号':<8} {'文件1的值':<15} {'文件2的值':<15}")
        print("-" * 40)
        for line_num, val1, val2 in differences:
            val1_str = f"{val1:.4f}" if val1 is not None else "缺失/无效"
            val2_str = f"{val2:.4f}" if val2 is not None else "缺失/无效"
            print(f"{line_num:<8} {val1_str:<15} {val2_str:<15}")
    print("=" * 70)

if __name__ == "__main__":
    main()