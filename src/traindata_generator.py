# traindata_generator.py
import os
import json
import random
import time
from typing import List, Dict

class SwiftDataGenerator:
    def __init__(self):
        pass

    def load_qa_pairs(self, file_path: str) -> List[Dict[str, str]]:
        qa_pairs = []
        print(f"正在加载文件: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return []
            
            try:
                if content.startswith('['):
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            if 'question' in item and 'answer' in item:
                                q = str(item['question']).strip()
                                a = str(item['answer']).strip()
                                if q and a:
                                    qa_pairs.append({'question': q, 'answer': a})
                else:
                    f.seek(0) 
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            item = json.loads(line)
                            if 'question' in item and 'answer' in item:
                                q = str(item['question']).strip()
                                a = str(item['answer']).strip()
                                if q and a:
                                    qa_pairs.append({'question': q, 'answer': a})
                        except json.JSONDecodeError:
                            continue
            except json.JSONDecodeError:
                print("文件格式解析失败")
                return []

        print(f"  成功加载 {len(qa_pairs)} 个有效问答对")
        return qa_pairs

    def generate_pairs_for_file(self, file_path: str, output_dir: str, max_global_count: int, current_count: List[int]) -> bool:
        qa_pairs = self.load_qa_pairs(file_path)
        
        if len(qa_pairs) < 2:
            print("  问答对数量不足，跳过。")
            return False

        all_answers = [qa['answer'] for qa in qa_pairs]
        all_questions = [qa['question'] for qa in qa_pairs]
        total_pairs = len(qa_pairs)

        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_dir, f"{name_without_ext}_swift_pairs.jsonl")
        os.makedirs(output_dir, exist_ok=True)

        written_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for qa in qa_pairs:
                if current_count[0] >= max_global_count:
                    return True 

                query = qa['question']
                pos_doc = qa['answer']

                # 1. 写入正例 (Label 1.0)
                pos_pair = {
                    "query": query,
                    "response": pos_doc,
                    "label": 1.0
                }
                f_out.write(json.dumps(pos_pair, ensure_ascii=False) + "\n")
                current_count[0] += 1
                written_count += 1

                if current_count[0] >= max_global_count:
                    break

                # 2. 写入负例 (Label 0.0)
                neg_doc = ""
                retry_count = 0
                while retry_count < 5:
                    neg_idx = random.randint(0, total_pairs - 1)
                    candidate_neg = all_answers[neg_idx]
                    
                    if candidate_neg != pos_doc and candidate_neg != query and len(candidate_neg) > 5:
                        neg_doc = candidate_neg
                        break
                    retry_count += 1
                
                if neg_doc:
                    neg_pair = {
                        "query": query,
                        "response": neg_doc,
                        "label": 0.0
                    }
                    f_out.write(json.dumps(neg_pair, ensure_ascii=False) + "\n")
                    current_count[0] += 1
                    written_count += 1

        print(f"  本文件生成 {written_count} 条 Pair 数据 → {output_file}")
        return current_count[0] >= max_global_count

    def run(self):
        print("\n--- SWIFT Embedding 数据生成器 (Pair+Label 模式) ---")
        
        while True:
            try:
                max_input = input("请输入期望生成的最大数据对总数 (默认 1000): ").strip()
                max_count = int(max_input) if max_input else 1000
                if max_count > 0:
                    break
                else:
                    print("请输入正整数")
            except ValueError:
                print("请输入有效的数字")

        test_dir = "../data/test/"
        if not os.path.exists(test_dir):
            print(f"错误: 目录不存在: {test_dir}")
            return

        qa_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jsonl', '.json'))])
        if not qa_files:
            print("未找到问答文件 (.json 或 .jsonl)")
            return

        print("\n可用文件：")
        for i, f in enumerate(qa_files, 1):
            print(f"  {i}. {f}")

        selection = input("\n请选择要处理的文件序号 (如 1,2,3 或全部留空): ").strip()
        
        selected_indices = []
        if not selection:
            selected_indices = list(range(len(qa_files)))
            print("已选择所有文件")
        else:
            try:
                selected_indices = [int(x) - 1 for x in selection.replace(" ", "").split(",")]
                selected_indices = [i for i in selected_indices if 0 <= i < len(qa_files)]
                if not selected_indices:
                    print("没有选择有效的文件")
                    return
            except ValueError:
                print("输入格式错误")
                return

        output_dir = "../data/training"
        current_count = [0] 
        
        start_time = time.time()
        
        for i in selected_indices:
            fname = qa_files[i]
            full_path = os.path.join(test_dir, fname)
            
            print(f"\n>>> 处理文件 [{i+1}/{len(selected_indices)}]: {fname}")
            
            stop_flag = self.generate_pairs_for_file(
                full_path, 
                output_dir, 
                max_count, 
                current_count
            )
            
            if stop_flag:
                print("\n已达到最大生成数量限制，停止处理。")
                break

        end_time = time.time()
        print(f"\n✅ 全部完成！")
        print(f"📊 共生成 {current_count[0]} 条训练数据")
        print(f"📂 保存路径：{output_dir}")
        print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")


def main():
    try:
        generator = SwiftDataGenerator()
        generator.run()
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()