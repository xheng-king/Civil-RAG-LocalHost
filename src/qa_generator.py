# qa_generator.py
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

    def generate_qa_pairs(self, text_chunk: str) -> List[dict]:
        """
        尝试从文本块中生成多个问答对。
        返回: 包含多个 {"question":..., "answer":...} 字典的列表。
              如果无法生成，返回空列表 []。
        """
        # 注意：f-string 中 JSON 的花括号需要双写 {{ }} 来转义
        prompt = f"""
请仔细阅读以下土木工程规范文本片段，并严格遵循以下要求生成问答对：

文本内容：
{text_chunk}

要求：
1. **最大化提取**：请挖掘文本中所有独立的专业知识点。不要只生成一个，要尽可能多地生成高质量的问答对。
2. **专业性问题**：问题必须基于专业实体（术语、构造、材料、设计规定、施工限制等）。**禁止**针对表格结构、公式符号、标点、排版、序号等非业务内容提问。
3. **独立性**：每个问题应当语义完整，不依赖上下文即可理解。
4. **答案规范**：答案必须严格摘抄或提炼自原文，客观严谨。
5. **无效处理**：如果文本中没有适合生成专业问答的内容（如纯页眉页脚、无意义字符），请仅返回字符串 "NULL"。
6. **输出格式**：
   - 若有内容，严格输出一个 **JSON 数组 (List)**，数组中包含多个对象。
   - 若无内容，返回字符串 "NULL"。

输出示例（有效时）：
[
    {{"question": "混凝土强度等级如何划分？", "answer": "混凝土强度等级应按立方体抗压强度标准值确定。"}},
    {{"question": "抗震等级分为几级？", "answer": "抗震等级分为一、二、三、四级。"}}
]

输出示例（无效时）：
NULL
"""

        try:
            response = self.client.chat.completions.create(
                model=llm,
                messages=[
                    {"role": "system", "content": "你是一个专业的土木工程知识库构建助手。你的任务是从给定的规范文本中提取尽可能多的专业问答对，并以JSON数组格式输出。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, # 低温度以保证准确性
                max_tokens=1500  # 增加 token 限制以容纳多个问答对
            )

            response_text = response.choices[0].message.content.strip()
            
            # 1. 预处理：去除 Markdown 代码块标记
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()

            # 2. 检查是否为 NULL 或空
            if not response_text or response_text.upper() == "NULL":
                return []
            
            # 3. 尝试解析 JSON
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # 解析失败，即模型输出了非标准 JSON，直接放弃，窗口滑动
                return []
            
            # 4. 验证数据结构
            qa_list = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'question' in item and 'answer' in item:
                        q = item.get('question', '').strip()
                        a = item.get('answer', '').strip()
                        # 简单的质量过滤：问题和答案不能太短
                        if len(q) > 5 and len(a) > 5:
                            qa_list.append({"question": q, "answer": a})
            elif isinstance(data, dict):
                # 兼容模型可能偶尔返回单个对象的情况
                if 'question' in data and 'answer' in data:
                     q = data.get('question', '').strip()
                     a = data.get('answer', '').strip()
                     if len(q) > 5 and len(a) > 5:
                         qa_list.append({"question": q, "answer": a})

            return qa_list

        except Exception as e:
            # API 错误或其他异常，返回空列表
            # print(f"  API Error: {e}")
            return []

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
                # 检查是否达到全局最大数量限制
                if current_count + successful_gen_count >= max_pairs:
                    print(f"  已达到最大问答对数量 {max_pairs}，停止处理。")
                    break

                end = start + window_size
                if end > total_chars:
                    end = total_chars
                    # 剩余内容过少则退出
                    if end - start < 50: 
                        break
                
                text_chunk = content[start:end]
                
                if not text_chunk.strip():
                    start += step
                    continue

                chunk_count += 1
                
                # 调用生成方法，获取列表
                qa_pairs = self.generate_qa_pairs(text_chunk)
                
                if qa_pairs:
                    # 批量写入
                    with open(self.output_file_path, 'a', encoding='utf-8') as f_out:
                        for qa in qa_pairs:
                            # 再次检查全局上限
                            if current_count + successful_gen_count >= max_pairs:
                                break
                            f_out.write(json.dumps(qa, ensure_ascii=False) + '\n')
                            successful_gen_count += 1
                    
                    print(f"  [Success] 窗口 #{chunk_count} 生成 {len(qa_pairs)} 个问答对。总计: {current_count + successful_gen_count}")
                else:
                    # 无有效内容，静默跳过
                    pass
                
                # 窗口滑动
                start = end + step

            print(f"文件 {file_path} 处理完成。本文件新增 {successful_gen_count} 个问答对。\n")
            return successful_gen_count

        except FileNotFoundError:
            print(f"错误: 文件未找到 {file_path}")
            return 0
        except UnicodeDecodeError:
            print(f"错误: 文件编码问题，无法读取 {file_path}。")
            return 0
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def run(self, file_paths: List[str], window_size: int, step: int, max_pairs: int):
        print(f"开始生成问答对，输出文件: {self.output_file_path}")
        print(f"参数: 窗口大小={window_size}, 步长={step}, 最大问答对数量={max_pairs}")
        
        # 清空输出文件
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
                window_size = int(input("请输入窗口大小（字符数，建议 1000-2000 以容纳更多内容）: ").strip())
                if window_size <= 0:
                    print("窗口大小必须是正整数，请重新输入。")
                    continue
                break
            except ValueError:
                print("无效输入，请输入整数。")
        
        while True:
            try:
                step = int(input("请输入步长（字符数，建议小于窗口大小以重叠覆盖，例如 800）: ").strip())
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
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()