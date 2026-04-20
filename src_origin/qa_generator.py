import os
import random
import json
from openai import OpenAI
from typing import List, Optional

from settings import base_url_set, llm

class QAPairGenerator:
    def __init__(self, api_key: Optional[str] = None):
        key_to_use = api_key or os.getenv("OPENAI_API_KEY")
        if not key_to_use:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量或在初始化时传入api_key参数")

        self.client = OpenAI(
            api_key=key_to_use,
            base_url=base_url_set
        )
        self.output_file_path = "../data/test/generated_qa.jsonl"

    def generate_qa_pair(self, text_chunk: str) -> Optional[dict]:
        prompt = f"""
请仔细阅读以下文本内容，并严格遵循以下要求生成一个问答对：

文本内容：
{text_chunk}

要求：
1.  **生成一个具体的问题 (Question)**：该问题应直接基于上述文本内容，能够通过阅读这段文字来回答。
2.  **生成一个准确的答案 (Answer)**：该答案应直接来自文本内容，是对所提问题的精确回答，要求简明扼要。
3.  **仅输出JSON格式**：请严格按照以下JSON格式输出，不要添加任何其他解释或文字。
4.  **问题和回答的精炼性**：不要使用“根据文本内容”、“请回答”等赘述性文字。

输出格式（严格遵守）：
{{"question": "基于文本的具体问题", "answer": "基于文本的具体答案"}}
"""

        try:
            response = self.client.chat.completions.create(
                model=llm, # 使用 settings.py 中配置的模型名称
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

    def process_file(self, file_path: str, window_size: int = 800, min_step: int = 4000, max_step: int = 6000):
        print(f"正在处理文件: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content:
                print(f"  文件 {file_path} 为空，跳过。")
                return

            start = 0
            chunk_count = 0
            successful_gen_count = 0
            
            total_chars = len(content)
            
            while start < total_chars:
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
                        print(f"    成功生成第 {successful_gen_count} 个问答对并保存。")
                    else:
                        print(f"    生成问答对失败。")
                else:
                    print(f"  跳过空白文本块 (位置: {start}-{end})。")
                
                step = random.randint(min_step, max_step)
                start = end + step

            print(f"文件 {file_path} 处理完成。成功生成 {successful_gen_count} 个问答对。\n")

        except FileNotFoundError:
            print(f"错误: 文件未找到 {file_path}")
        except UnicodeDecodeError:
            print(f"错误: 文件编码问题，无法读取 {file_path}。请确保文件为UTF-8编码。")
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    def run(self, file_paths: List[str]):
        print(f"开始生成问答对，输出文件: {self.output_file_path}")
        with open(self.output_file_path, 'w', encoding='utf-8') as f:
            pass

        for file_path in file_paths:
            abs_file_path = os.path.abspath(file_path)
            self.process_file(abs_file_path)

def main():
    try:
        generator = QAPairGenerator()
        
        txt_dir = "../data/src_files/"
        if not os.path.exists(txt_dir):
            print(f"目录 {txt_dir} 不存在")
            return
        
        all_files = os.listdir(txt_dir)
        txt_files = [f for f in all_files if f.lower().endswith(('.txt', '.md'))]
        sorted_files = sorted(txt_files)
        
        if not sorted_files:
            print("src_files 目录中没有找到任何 .txt 或 .md 文件")
            return

        print("\nsrc_files 目录中的文件:")
        for i, filename in enumerate(sorted_files, 1):
            print(f"  {i}. {filename}")
        
        if not sorted_files:
            print("没有找到可处理的文件，跳过生成")
            return
        
        print(f"\n请输入要处理的文件序号，支持单选或多选")
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
        
        generator.run(file_paths)
        print("\n所有选定的文件处理完成！问答对已保存到 generated_qa.jsonl")

    except ValueError as e:
        print(f"初始化错误: {e}")
        print("请确保已设置 OPENAI_API_KEY 环境变量。")
    except Exception as e:
        print(f"执行过程中发生未预期的错误: {e}")

if __name__ == "__main__":
    main()