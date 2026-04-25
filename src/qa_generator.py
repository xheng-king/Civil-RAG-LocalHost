import os
import json
from openai import OpenAI
from typing import List, Optional

from settings import llm, llm_base_url, llm_API_key

class QAPairGenerator:
    def __init__(self, api_key: Optional[str] = None):
        key_to_use = api_key or llm_API_key
        if not key_to_use:
            raise ValueError("settings.py 中的 llm_API_key 未设置")
        
        self.client = OpenAI(
            api_key=key_to_use,
            base_url=llm_base_url
        )
        self.output_file_path = "../data/test/generated_qa.jsonl"

    def generate_qa_pair(self, text_chunk: str) -> Optional[dict]:
        prompt = f"""
请仔细阅读以下文本内容，并严格遵循以下要求生成一个问答对：

文本内容：
{text_chunk}

要求：
1.  生成一个具体的问题 (Question)：该问题应直接基于上述文本内容，能够通过阅读这段文字来回答。
2.  生成一个准确的答案 (Answer)：该答案应直接来自文本内容，是对所提问题的精确回答，要求简明扼要。
3.  仅输出JSON格式：请严格按照以下JSON格式输出，不要添加任何其他解释或文字。
4.  问题和回答的精炼性与完整性：不要使用“根据文本内容”、“请回答”等赘述性文字，文本中语义不全的信息不要作为问答对。

输出格式（严格遵守）：
{{"question": "基于文本的具体问题", "answer": "基于文本的具体答案"}}
"""

        try:
            response = self.client.chat.completions.create(
                model=llm,
                messages=[
                    {"role": "system", "content": "你是一个专业的问答对生成助手。你会根据给定的文本内容，生成一个准确、具体的问答对，并严格按照JSON格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            response_text = response.choices[0].message.content.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            try:
                qa_dict = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"  错误: 无法解析模型返回的JSON: {response_text[:100]}... (错误详情: {e})")
                return None
            
            if isinstance(qa_dict, dict) and 'question' in qa_dict and 'answer' in qa_dict:
                return qa_dict
            else:
                print(f"  警告: 模型返回的格式不符合要求: {response_text}")
                return None

        except Exception as e:
            print(f"  错误: 调用模型生成问答对时出错: {e}")
            return None

    def process_file(self, file_path: str, window_size: int, step: int, max_pairs: int, current_count: int) -> int:
        print(f"正在处理文件: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content:
                print(f"  文件 {file_path} 为空，跳过。")
                return 0

            start = 0
            chunk_count = 0
            successful_gen_count = 0
            
            total_chars = len(content)
            
            while start < total_chars:
                if current_count + successful_gen_count >= max_pairs:
                    print(f"  已达到最大问答对数量 {max_pairs}，停止处理当前文件。")
                    break

                end = start + window_size
                if end > total_chars:
                    end = total_chars
                    if end - start < 10:
                        break
                
                text_chunk = content[start:end]
                
                if text_chunk.strip():
                    chunk_count += 1
                    print(f"  处理第 {chunk_count} 个文本块 (位置: {start}-{end})...")

                    qa_pair = self.generate_qa_pair(text_chunk)
                    if qa_pair:
                        with open(self.output_file_path, 'a', encoding='utf-8') as f_out:
                            f_out.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                        successful_gen_count += 1
                        print(f"    成功生成第 {current_count + successful_gen_count} 个问答对并保存。")
                    else:
                        print(f"    生成问答对失败。")
                else:
                    print(f"  跳过空白文本块 (位置: {start}-{end})。")
                
                start = end + step

            print(f"文件 {file_path} 处理完成。本文件成功生成 {successful_gen_count} 个问答对。\n")
            return successful_gen_count

        except FileNotFoundError:
            print(f"错误: 文件未找到 {file_path}")
            return 0
        except UnicodeDecodeError:
            print(f"错误: 文件编码问题，无法读取 {file_path}。请确保文件为UTF-8编码。")
            return 0
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return 0

    def run(self, file_paths: List[str], window_size: int, step: int, max_pairs: int):
        print(f"开始生成问答对，输出文件: {self.output_file_path}")
        print(f"参数: 窗口大小={window_size}, 步长={step}, 最大问答对数量={max_pairs}")
        
        with open(self.output_file_path, 'w', encoding='utf-8') as f:
            pass

        total_generated = 0
        for file_path in file_paths:
            if total_generated >= max_pairs:
                print(f"已达到最大问答对数量 {max_pairs}，停止处理后续文件。")
                break
            abs_file_path = os.path.abspath(file_path)
            generated = self.process_file(abs_file_path, window_size, step, max_pairs, total_generated)
            total_generated += generated

        print(f"\n全部处理完成。共成功生成 {total_generated} 个问答对。")

def main():
    try:
        generator = QAPairGenerator()
        
        txt_dir = "../data/src_files/"
        if not os.path.exists(txt_dir):
            print(f"目录 {txt_dir} 不存在")
            return
        
        print("\n--- 参数设置 ---")
        while True:
            try:
                window_size = int(input("请输入窗口大小（字符数，例如 800）: ").strip())
                if window_size <= 0:
                    print("窗口大小必须是正整数，请重新输入。")
                    continue
                break
            except ValueError:
                print("无效输入，请输入整数。")
        
        while True:
            try:
                step = int(input("请输入步长（字符数，例如 5000）: ").strip())
                if step <= 0:
                    print("步长必须是正整数，请重新输入。")
                    continue
                break
            except ValueError:
                print("无效输入，请输入整数。")
        
        while True:
            try:
                max_pairs = int(input("请输入最大问答对数量（例如 5000）: ").strip())
                if max_pairs <= 0:
                    print("最大数量必须是正整数，请重新输入。")
                    continue
                break
            except ValueError:
                print("无效输入，请输入整数。")
        
        print(f"\n参数确认：窗口大小 = {window_size}，步长 = {step}，最大问答对数量 = {max_pairs}")
        
        all_files = os.listdir(txt_dir)
        txt_files = [f for f in all_files if f.lower().endswith(('.txt', '.md'))]
        sorted_files = sorted(txt_files)
        
        if not sorted_files:
            print("src_files 目录中没有找到任何 .txt 或 .md 文件")
            return

        print("\nsrc_files 目录中的文件:")
        for i, filename in enumerate(sorted_files, 1):
            print(f"  {i}. {filename}")
        
        print("\n请输入要处理的文件序号，支持单选或多选")
        print("例如: 输入 '1' 表示选择第1个文件")
        print("例如: 输入 '1,2,4' 表示选择第1、2、4个文件")
        
        selection_str = input("请选择文件序号: ").strip()
        
        try:
            selections = selection_str.replace(' ', '').split(',')
            selected_indices = []
            
            for s in selections:
                num = int(s)
                if 1 <= num <= len(sorted_files):
                    selected_indices.append(num - 1)
                else:
                    print(f"警告: 选择 {num} 超出范围 (1-{len(sorted_files)})，已忽略")
            
            unique_selected_indices = list(set(selected_indices))
        except ValueError:
            print("输入格式错误，请输入数字，多个数字用逗号分隔")
            return

        if not unique_selected_indices:
            print("没有选择有效的文件，跳过生成")
            return
        
        selected_files = [sorted_files[i] for i in unique_selected_indices]
        print(f"\n已选择以下文件: {selected_files}")
        
        confirm = input("\n确认开始处理? (y/N): ")
        if confirm.lower() != 'y':
            print("操作已取消。")
            return

        file_paths = [os.path.abspath(f"../data/src_files/{filename}") for filename in selected_files]
        
        generator.run(file_paths, window_size, step, max_pairs)
        print("\n处理完成！问答对已保存到 generated_qa.jsonl")

    except ValueError as e:
        print(f"初始化错误: {e}")
        print("请确保 settings.py 中的 llm_API_key 已正确设置。")
    except Exception as e:
        print(f"执行过程中发生未预期的错误: {e}")

if __name__ == "__main__":
    main()