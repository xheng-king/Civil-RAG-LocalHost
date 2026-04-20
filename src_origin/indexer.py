# src/indexer.py
import os
import re
import chromadb
import csv
from openai import OpenAI
from typing import List
from database_manager import DatabaseManager

class QwenIndexer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        self.chroma_client = chromadb.PersistentClient(path="../data/vectorstore")
        self.db_manager = DatabaseManager()

    def blocks(self, text: str) -> List[str]:
        clause_pattern = r'\n\s*(?:\d+(?:\.\d+)+~)?\d+(?:\.\d+)+\s+'
        clauses_found = list(re.finditer(clause_pattern, text))
        
        if not clauses_found:
            return [text.strip()] if text.strip() else []

        chunks = []
        start = 0
        
        for match in clauses_found:
            clause_end = match.end()
            chunk_content = text[start : clause_end].strip()
            
            if chunk_content:
                chunks.append(chunk_content)
            
            start = clause_end
        
        if start < len(text):
            remaining_content = text[start:].strip()
            if remaining_content:
                chunks.append(remaining_content)
        
        return chunks

    def connect(self, str1: str, str2: str) -> str:
        if not str1:
            return str2
        if not str2:
            return str1
        return str1 + "\n" + str2

    def cut_string(self, s: str, start: int, end: int) -> str:
        return s[start:end]

    def structural_chunk(self, text: str, min_chunk_size: int = 512, max_chunk_size: int = 2048) -> List[str]:
        initial_chunks = self.blocks(text)
        final_chunks = []
        current_chunk = ""

        for block_i in initial_chunks:
            while len(current_chunk) > max_chunk_size:
                part = self.cut_string(current_chunk, 0, max_chunk_size)
                final_chunks.append(part)
                current_chunk = self.cut_string(current_chunk, max_chunk_size, len(current_chunk))

            if len(current_chunk) == max_chunk_size:
                final_chunks.append(current_chunk)
                current_chunk = ""

            potential_chunk = self.connect(current_chunk, block_i)

            if len(potential_chunk) <= max_chunk_size:
                current_chunk = potential_chunk

                if len(current_chunk) >= min_chunk_size:
                    final_chunks.append(current_chunk)
                    current_chunk = ""
            else:
                if current_chunk != "":
                    final_chunks.append(current_chunk)
                    current_chunk = block_i
                else:
                    long_part = self.cut_string(block_i, 0, max_chunk_size)
                    remaining_part = self.cut_string(block_i, max_chunk_size, len(block_i))
                    final_chunks.append(long_part)
                    current_chunk = remaining_part

        while len(current_chunk) > max_chunk_size:
            part = self.cut_string(current_chunk, 0, max_chunk_size)
            final_chunks.append(part)
            current_chunk = self.cut_string(current_chunk, max_chunk_size, len(current_chunk))
        
        if current_chunk.strip():
            final_chunks.append(current_chunk)
        
        return final_chunks

    def read_and_chunk_file(self, file_path: str, min_chunk_size: int = 512, max_chunk_size: int = 2048) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        print("read_and_chunk_file:文件打开成功，并成功读取内容")
        
        segments = self.structural_chunk(content, min_chunk_size, max_chunk_size)
        print("文件按条款结构分块成功")
        print(f"最小分块大小: {min_chunk_size} 字符, 最大分块大小: {max_chunk_size} 字符")
        print(f"共生成 {len(segments)} 个文本块")
        
        for i in range(min(3, len(segments))):
            print(f"  块 {i+1}: {len(segments[i])} 字符")
        
        if len(segments) > 3:
            print(f"  ... 还有 {len(segments) - 3} 个块")
        
        return segments
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            print(f"正在为文本块 {i+1}/{len(texts)} 生成 Qwen 嵌入... (大小: {len(text)} 字符)")
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-v2",# 嵌入维度:1536
                    input=text
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
                print(f"  成功，嵌入维度: {len(embedding)}")
            except Exception as e:
                print(f"  生成嵌入时出错: {e}")
                raise e
        
        return embeddings
    
    def index_single_file_to_collection(self, file_path: str, collection_name: str, min_chunk_size: int = 512, max_chunk_size: int = 2048):
        print(f"\n正在处理文件: {os.path.basename(file_path)}")
        print(f"使用默认最小分块大小: {min_chunk_size} 字符, 最大分块大小: {max_chunk_size} 字符")
        
        segments = self.read_and_chunk_file(file_path, min_chunk_size, max_chunk_size)
        print("文件按结构分块成功")
        if not segments:
            print(f"文件 {file_path} 为空或没有有效内容，跳过索引")
            return
        
        embeddings = self.create_embeddings(segments)
        
        collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        ids = [f"{base_filename}_doc_{i}" for i in range(len(segments))]
        metadatas = [{
            "source": os.path.basename(file_path),
            "segment_number": i+1,
            "file": base_filename,
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": max_chunk_size,
            "length": len(segments[i])
        } for i in range(len(segments))]
        
        collection.add(
            documents=segments,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        print(f"成功将 {len(segments)} 个文档从 {os.path.basename(file_path)} 索引到集合 '{collection_name}'")

        # --- 修改：写入 CSV 文件 ---
        chunk_details_csv_path = "../chunk_details.csv"
        file_summary_csv_path = "../file_summary.csv"

        # 1. 写入 chunk_details.csv
        chunk_details_file_exists = os.path.isfile(chunk_details_csv_path)
        with open(chunk_details_csv_path, 'a', newline='', encoding='utf-8') as chunk_csvfile:
            chunk_fieldnames = ['file_name', 'chunk_index', 'length']
            chunk_writer = csv.DictWriter(chunk_csvfile, fieldnames=chunk_fieldnames)

            if not chunk_details_file_exists:
                chunk_writer.writeheader()
            
            for i, segment in enumerate(segments):
                chunk_writer.writerow({
                    'file_name': os.path.basename(file_path),
                    'chunk_index': i,
                    'length': len(segment),
                })

        # 2. 写入 file_summary.csv
        total_chunks = len(segments)
        avg_length = sum(len(seg) for seg in segments) / total_chunks if total_chunks > 0 else 0

        file_summary_file_exists = os.path.isfile(file_summary_csv_path)
        with open(file_summary_csv_path, 'a', newline='', encoding='utf-8') as summary_csvfile:
            summary_fieldnames = ['file_name', 'total_chunks', 'average_chunk_length']
            summary_writer = csv.DictWriter(summary_csvfile, fieldnames=summary_fieldnames)

            if not file_summary_file_exists:
                summary_writer.writeheader()
            
            summary_writer.writerow({
                'file_name': os.path.basename(file_path),
                'total_chunks': total_chunks,
                'average_chunk_length': round(avg_length, 2) # 保留两位小数
            })
