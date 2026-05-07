#!/usr/bin/env python3
"""
remove_duplicate_pairs.py

功能：从 test_easy_lib.jsonl 中删除所有在 test_easy.jsonl 中出现的问答对。
使用方式：直接运行脚本（需保证两个文件位于当前目录下），或修改脚本中的文件名变量。
"""

import json
import os
import shutil

# 配置文件名（可修改）
LIB_FILE = "test_easy_lib.jsonl"
TEST_FILE = "test_easy.jsonl"
BACKUP_SUFFIX = ".bak"  # 备份文件后缀


def load_pairs(file_path: str) -> set:
    """
    从 JSONL 文件中加载所有 (question, answer) 元组。
    返回一个集合，每个元素为 (question_stripped, answer_stripped)。
    """
    pairs = set()
    if not os.path.exists(file_path):
        print(f"警告：文件 {file_path} 不存在，将作为空集合处理。")
        return pairs

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # 提取 question 和 answer，去除首尾空格
                q = data.get('question', '').strip()
                a = data.get('answer', '').strip()
                if q and a:  # 只保留有效问答对
                    pairs.add((q, a))
                else:
                    print(f"警告：文件 {file_path} 第 {line_num} 行缺少 question 或 answer，已跳过")
            except json.JSONDecodeError as e:
                print(f"错误：文件 {file_path} 第 {line_num} 行 JSON 解析失败：{e}")
    return pairs


def filter_lib_file(lib_file: str, easy_pairs: set, backup: bool = True):
    """过滤 lib_file，保留不在 easy_pairs 中的条目，并可选地创建备份。"""
    # 读取所有行并判断
    kept_lines = []
    with open(lib_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            try:
                data = json.loads(line_stripped)
                q = data.get('question', '').strip()
                a = data.get('answer', '').strip()
                # 如果该问答对不在 easy_pairs 中，则保留
                if (q, a) not in easy_pairs:
                    kept_lines.append(line.rstrip('\n'))  # 保留原始换行符
                else:
                    print(f"删除重复条目（第 {line_num} 行）：{data}")
            except json.JSONDecodeError as e:
                print(f"错误：{lib_file} 第 {line_num} 行解析失败，保留原行：{e}")
                kept_lines.append(line.rstrip('\n'))

    if not kept_lines:
        print("警告：过滤后没有剩余条目，将清空文件。")

    # 创建备份
    if backup and os.path.exists(lib_file):
        backup_file = lib_file + BACKUP_SUFFIX
        shutil.copy2(lib_file, backup_file)
        print(f"已创建备份：{backup_file}")

    # 写回原文件
    with open(lib_file, 'w', encoding='utf-8') as f:
        for line in kept_lines:
            f.write(line + '\n')
    print(f"已更新文件：{lib_file}，保留了 {len(kept_lines)} 条记录。")


def main():
    # 检查两个文件是否存在（lib 文件必须存在，easy 文件可选）
    if not os.path.exists(LIB_FILE):
        print(f"错误：找不到文件 {LIB_FILE}，请确保脚本与文件在同一目录下，或修改脚本中的 LIB_FILE 变量。")
        return

    if not os.path.exists(TEST_FILE):
        print(f"错误：找不到文件 {TEST_FILE}，无法进行去重。")
        return

    print(f"正在加载 {TEST_FILE} 中的问答对...")
    easy_pairs = load_pairs(TEST_FILE)
    print(f"从 {TEST_FILE} 中加载了 {len(easy_pairs)} 个唯一问答对。")

    print(f"正在过滤 {LIB_FILE}...")
    filter_lib_file(LIB_FILE, easy_pairs, backup=True)

    print("操作完成。")


if __name__ == "__main__":
    main()