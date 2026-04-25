# image_captioner.py
import re
import json
import base64
from pathlib import Path
from openai import OpenAI

# 从 settings.py 导入视觉模型相关配置
from settings import vllm, vllm_base_url, vllm_API_key

class ImageCaptioner:
    def __init__(self):
        # 直接使用 settings 中配置的视觉模型 API Key 和 Base URL
        api_key = vllm_API_key
        if not api_key:
            raise ValueError("settings.py 中的 vllm_API_key 未设置。")
        
        # 使用 settings 中配置的 vllm_base_url（通常为视觉模型专用端点）
        self.client = OpenAI(
            api_key=api_key,
            base_url=vllm_base_url
        )
        
        # 设置项目根目录和缓存目录
        self.project_root = Path(__file__).parent.parent.resolve()
        self.cache_dir = self.project_root / 'cache'
        self.src_files_dir = self.project_root / 'data' / 'src_files'
        
        # 创建必要的目录
        self.cache_dir.mkdir(exist_ok=True)

    def encode_image(self, image_path):
        """将图片文件编码为base64字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def call_vision_model(self, image_path):
        """调用视觉模型生成图片描述"""
        base64_image = self.encode_image(image_path)
        
        prompt = (
            "请用中文简洁明确地描述这张图片的主要内容。"
            "如果是工程图纸或图表，请重点说明其表达的技术要点、结构特征或计算示意。"
            "如果是公式推导图，请概述其涉及的核心概念。"
            "如果是表格截图，请概括其主题和关键数据。"
            "输出应精炼，突出关键信息，避免冗余描述。"
        )

        response = self.client.chat.completions.create(
            model=vllm,  # 使用 settings.vllm，例如 "qwen3-vl-flash"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.7,
        )
        
        description = response.choices[0].message.content.strip()
        return description

    def process_markdown_file(self, md_file_path):
        """处理单个Markdown文件"""
        md_file_path = Path(md_file_path).resolve()
        if not md_file_path.exists():
            print(f"文件不存在: {md_file_path}")
            return

        print(f"\n开始处理文件: {md_file_path.name}")
        
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        img_pattern = r'(!\[([^\]]*)\]\(([^)]+\.(?:jpg|jpeg|png|gif|bmp|webp))\))'
        matches = list(re.finditer(img_pattern, content, re.IGNORECASE))
        
        if not matches:
            print(f"  - 未找到图片链接，跳过处理。")
            return

        total_images = len(matches)
        print(f"  - 发现 {total_images} 个图片链接")

        cache_file = self.cache_dir / f"caption_progress_{md_file_path.name}.json"
        processed_links = set()
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as cf:
                    cached_data = json.load(cf)
                    processed_links = set(cached_data.get('completed', []))
                    print(f"  - 从缓存加载进度，已处理 {len(processed_links)} 个链接")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  - 加载缓存失败: {e}, 将重新开始处理。")
                processed_links = set()

        updated_content = content
        any_new_caption_added = False
        
        for i, match in enumerate(matches):
            full_match = match.group(0)
            img_src = match.group(3).strip()

            print(f"    处理第 {i+1}/{total_images} 个图片: {img_src}")

            if img_src in processed_links:
                print(f"      - 已在缓存中，跳过。")
                continue

            img_abs_path = self.src_files_dir / img_src
            
            if not img_abs_path.exists():
                print(f"      - 错误: 图片文件不存在: {img_abs_path}")
                continue

            try:
                print(f"      - 正在调用视觉模型...")
                caption_text = self.call_vision_model(img_abs_path)
                
                comment_block = f"\n<!-- 图片描述: {caption_text} -->\n"
                updated_content = updated_content.replace(full_match, full_match + comment_block, 1)
                
                processed_links.add(img_src)
                
                with open(cache_file, 'w', encoding='utf-8') as cf:
                    json.dump({"completed": list(processed_links)}, cf, ensure_ascii=False, indent=2)
                
                print(f"      - 成功生成描述并附加到文档。")
                any_new_caption_added = True
                
            except Exception as e:
                print(f"      - 生成描述时出错: {e}")
                print("      - 跳过此图片，继续处理下一个。")
                continue

        if any_new_caption_added:
            print(f"  - 写入更新后的文件: {md_file_path}")
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            if cache_file.exists():
                cache_file.unlink()
                print(f"  - 删除进度缓存: {cache_file.name}")
        
        print(f"文件 {md_file_path.name} 处理完成。")

    def process_multiple_files(self, file_paths):
        """批量处理多个Markdown文件"""
        for path in file_paths:
            self.process_markdown_file(path)
            print("-" * 40)