# image_captioner.py
import os
import re
import json
import base64
from pathlib import Path
from openai import OpenAI

# 导入 settings.py 中的配置
from settings import base_url_set, vllm

class ImageCaptioner:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("环境变量 OPENAI_API_KEY 未设置。")
        
        # 使用 settings.py 中配置的 base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url_set
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
        """调用Qwen视觉模型生成图片描述"""
        base64_image = self.encode_image(image_path)
        
        prompt = (
            "请用中文简洁明确地描述这张图片的主要内容。"
            "如果是工程图纸或图表，请重点说明其表达的技术要点、结构特征或计算示意。"
            "如果是公式推导图，请概述其涉及的核心概念。"
            "如果是表格截图，请概括其主题和关键数据。"
            "输出应精炼，突出关键信息，避免冗余描述。"
        )

        response = self.client.chat.completions.create(
            # 使用 settings.py 中配置的模型名称
            model=vllm,  # 视觉语言模型
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}" # 使用Data URL格式
                            }
                        },
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.7,  # 可选参数，控制输出随机性
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
        
        # 读取Markdown文件内容
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找所有图片链接 ![](path) 或 ![alt](path)
        # 支持相对路径 images/xxx.jpg 或 ./images/xxx.jpg
        # 也支持类似您示例中的路径 images/uuid.jpg
        img_pattern = r'(!\[([^\]]*)\]\(([^)]+\.(?:jpg|jpeg|png|gif|bmp|webp))\))'
        matches = list(re.finditer(img_pattern, content, re.IGNORECASE))
        
        if not matches:
            print(f"  - 未找到图片链接，跳过处理。")
            return

        total_images = len(matches)
        print(f"  - 发现 {total_images} 个图片链接")

        # 加载或初始化进度缓存
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
            full_match = match.group(0) # 整个 ![]() 匹配项
            img_src = match.group(3).strip() # 图片路径部分

            print(f"    处理第 {i+1}/{total_images} 个图片: {img_src}")

            # 检查是否已处理过
            if img_src in processed_links:
                print(f"      - 已在缓存中，跳过。")
                continue

            # 构建图片的绝对路径
            # 假设图片相对于md文件所在目录（即 src_files/ ）寻找
            img_abs_path = self.src_files_dir / img_src
            
            if not img_abs_path.exists():
                print(f"      - 错误: 图片文件不存在: {img_abs_path}")
                continue

            try:
                # 调用视觉模型生成描述
                print(f"      - 正在调用视觉模型...")
                caption_text = self.call_vision_model(img_abs_path)
                
                # 生成注释形式的描述，附加在原链接后面
                # 使用HTML注释格式，这样不会在渲染时显示，但保留在源码中
                comment_block = f"\n<!-- 图片描述: {caption_text} -->\n"
                
                # 将描述追加到原链接后面
                updated_content = updated_content.replace(full_match, full_match + comment_block, 1)
                
                # 记录已处理的链接
                processed_links.add(img_src)
                
                # 更新缓存文件
                with open(cache_file, 'w', encoding='utf-8') as cf:
                    json.dump({"completed": list(processed_links)}, cf, ensure_ascii=False, indent=2)
                
                print(f"      - 成功生成描述并附加到文档。")
                any_new_caption_added = True
                
            except Exception as e:
                print(f"      - 生成描述时出错: {e}")
                print("      - 跳过此图片，继续处理下一个。")
                continue # 继续处理下一个图片，不中断整个文件

        # 如果有任何新的描述被添加，则写回文件
        if any_new_caption_added:
            print(f"  - 写入更新后的文件: {md_file_path}")
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            # 处理完一个文件后，删除对应的缓存记录
            if cache_file.exists():
                cache_file.unlink()
                print(f"  - 删除进度缓存: {cache_file.name}")
        
        print(f"文件 {md_file_path.name} 处理完成。")

    def process_multiple_files(self, file_paths):
        """批量处理多个Markdown文件"""
        for path in file_paths:
            self.process_markdown_file(path)
            print("-" * 40) # 分隔符，便于查看日志