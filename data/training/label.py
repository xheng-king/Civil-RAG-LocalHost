# label.py
# 为初始训练集进行交互式人工标注
import json
import os
from pathlib import Path
from collections import defaultdict

def list_jsonl_files():
    """列出当前目录下所有.jsonl文件"""
    files = [f for f in os.listdir('.') if f.endswith('.jsonl')]
    if not files:
        print("当前目录下没有找到.jsonl文件。")
        return None
    print("可用的.jsonl文件：")
    for idx, f in enumerate(files, 1):
        print(f"{idx}. {f}")
    while True:
        try:
            choice = int(input("请选择文件编号: "))
            if 1 <= choice <= len(files):
                return files[choice-1]
            else:
                print(f"请输入1~{len(files)}之间的数字。")
        except ValueError:
            print("请输入有效的数字。")

def load_data(filepath):
    """读取jsonl文件，返回按query分组的字典 {query: [(response, original_label, line_num)]}"""
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
                label = data.get('label')
                if query is None or response is None or label is None:
                    print(f"警告：第{line_num}行缺少query/response/label字段，已跳过。")
                    continue
                groups[query].append((response, label, line_num))
            except json.JSONDecodeError:
                print(f"警告：第{line_num}行不是合法JSON，已跳过。")
    return groups

def load_labeled_queries(output_file):
    """读取已标注输出文件，返回其中所有出现过的query集合（只要query存在，即认为已标注）"""
    labeled_queries = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
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

def get_user_label(current_label):
    """
    交互获取用户输入的新标签。
    返回: (new_label, skip)   skip=True 表示跳过该response；否则new_label为0或1
    """
    prompt = f"当前标签: {current_label}  -> 输入 0/1 修改，按回车保持，输入 r 跳过: "
    user_input = input(prompt).strip().lower()
    if user_input == 'r':
        return None, True   # 跳过
    if user_input == '':
        # 保持原标签，转换为0或1
        new_label = 1 if float(current_label) >= 0.5 else 0
        return new_label, False
    # 尝试解析为数字
    try:
        val = float(user_input)
        new_label = 1 if val >= 0.5 else 0
        return new_label, False
    except ValueError:
        print("输入无效，将保持原标签（转换为0/1）。")
        new_label = 1 if float(current_label) >= 0.5 else 0
        return new_label, False

def main():
    jsonl_file = list_jsonl_files()
    if not jsonl_file:
        return
    
    # 按query分组加载原始数据
    groups = load_data(jsonl_file)
    if not groups:
        print("文件中没有有效数据。")
        return
    
    # 输出文件
    base_name = Path(jsonl_file).stem
    output_file = f"{base_name}_labeled.jsonl"
    
    # 已标注过的query集合（只要query在输出文件中出现过，就跳过整个query）
    labeled_queries = load_labeled_queries(output_file)
    if labeled_queries:
        print(f"发现已标注的query: {len(labeled_queries)} 个，将跳过它们。")
    
    # 待标注的query列表（未在labeled_queries中的）
    pending_queries = [(q, resp_list) for q, resp_list in groups.items() if q not in labeled_queries]
    if not pending_queries:
        print("所有query均已标注完毕，无需操作。")
        return
    
    print(f"本次需要标注 {len(pending_queries)} 个query。")
    
    # 打开输出文件（追加模式）
    with open(output_file, 'a', encoding='utf-8') as out_f:
        for idx, (query, responses) in enumerate(pending_queries, 1):
            print(f"\n========== Query {idx}/{len(pending_queries)} ==========")
            print(f"Query: {query}")
            input("按任意键开始处理该query下的responses...")
            
            for resp, orig_label, line_num in responses:
                print(f"\n  Response (来自原文件第{line_num}行):")
                print(f"  {resp}")
                new_label, skip = get_user_label(orig_label)
                if skip:
                    print("  已跳过该response，不会写入输出文件。")
                    continue
                
                # 写入新的三元组
                new_entry = {
                    "query": query,
                    "response": resp,
                    "label": new_label
                }
                out_f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                out_f.flush()
                print(f"  已写入标签: {new_label}")
            
            print(f"Query '{query[:50]}...' 处理完毕。")
    
    print(f"\n标注完成！结果已保存至: {output_file}")

if __name__ == "__main__":
    main()