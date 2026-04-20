import os
import json
import numpy as np
from database_manager import DatabaseManager
from indexer import QwenIndexer
from rag_interface import interactive_query
from image_captioner import ImageCaptioner
from qa_generator import main as run_qa_generation

def get_txt_files():
    txt_dir = "../data/src_files/"
    if not os.path.exists(txt_dir):
        print(f"目录 {txt_dir} 不存在")
        return []
    
    txt_files = [f for f in os.listdir(txt_dir) if f.lower().endswith('.txt') or f.lower().endswith('.md')]
    return sorted(txt_files)

def display_txt_files():
    txt_files = get_txt_files()
    
    if not txt_files:
        print("src_files 目录中没有找到任何文件")
        return []
    
    print("\nsrc_files 目录中的文件:")
    for i, filename in enumerate(txt_files, 1):
        print(f"  {i}. {filename}")
    
    return txt_files

def parse_selection(selection_str, max_count):
    try:
        selections = selection_str.replace(' ', '').split(',')
        selected_indices = []
        
        for s in selections:
            num = int(s)
            if 1 <= num <= max_count:
                selected_indices.append(num - 1)
            else:
                print(f"警告: 选择 {num} 超出范围 (1-{max_count})，已忽略")
        
        return list(set(selected_indices))
    except ValueError:
        print("输入格式错误，请输入数字，多个数字用逗号分隔")
        return []

def export_collection_content(db_manager, collection_name):
    try:
        collection = db_manager.client.get_collection(name=collection_name)
        
        result = collection.get(include=['documents', 'metadatas', 'embeddings'])
        
        export_data = []
        for doc, meta, emb in zip(result['documents'], result['metadatas'], result['embeddings']):
            if isinstance(emb, np.ndarray):
                emb_list = emb.tolist()
            elif emb is None:
                emb_list = []
            else:
                emb_list = list(emb) if emb else []
            
            export_item = {
                'document': doc,
                'metadata': meta,
                'embedding_first_5_dims': emb_list[:5] if emb_list else []
            }
            export_data.append(export_item)
        
        export_dir = "../data/exported_collections/"
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        export_file_path = os.path.join(export_dir, f"{collection_name}.json")
        
        with open(export_file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"集合 '{collection_name}' 的内容已导出到: {export_file_path}")
        print(f"导出了 {len(export_data)} 个文本-向量对")
        
    except Exception as e:
        print(f"导出集合内容时出错: {e}")

def handle_image_captioning():
    """处理Markdown文件中的图片，生成描述并附加到文本"""
    print("\n--- Markdown文件图像处理 ---")
    
    txt_files = display_txt_files()
    if not txt_files:
        print("没有找到可处理的 .txt 或 .md 文件，跳过处理")
        return
    
    # 筛选出 .md 文件
    md_files = [f for f in txt_files if f.lower().endswith('.md') or f.lower().endswith('.txt')]
    if not md_files:
        print("src_files 目录中没有找到 .md 文件，跳过处理")
        return

    print("\nsrc_files 目录中的 .md 文件:")
    for i, filename in enumerate(md_files, 1):
        print(f"  {i}. {filename}")

    print(f"\n请输入要处理的 .md 文件序号，支持单选或多选")
    print("例如: 输入 '1' 表示选择第1个文件")
    print("例如: 输入 '1,2,4' 表示选择第1、2、4个文件")
    
    selection_str = input("请选择文件序号: ").strip()
    selected_indices = parse_selection(selection_str, len(md_files))
    
    if not selected_indices:
        print("没有选择有效的文件，跳过处理")
        return
    
    selected_md_files = [md_files[i] for i in selected_indices]
    print(f"\n已选择以下 .md 文件: {selected_md_files}")

    confirm = input("\n确认开始处理? (y/N): ")
    if confirm.lower() != 'y':
        print("操作已取消。")
        return

    # 获取文件的绝对路径列表
    file_paths = [os.path.abspath(f"../data/src_files/{filename}") for filename in selected_md_files]

    # 初始化并启动图片描述器
    captioner = ImageCaptioner()
    try:
        captioner.process_multiple_files(file_paths)
        print("\n所有选定的 .md 文件的图片处理完成！")
    except ValueError as ve:
        print(f"\n错误: {ve}")
        print("请确保已设置 OPENAI_API_KEY 环境变量，并且模型名正确（例如 qwen-vl-max）")
    except Exception as e:
        print(f"\n处理过程中出错: {e}")


def main():
    print("="*60)
    print("Qwen RAG 系统 - 统一管理界面")
    print("="*60)
    
    while True:
        print("\n请选择功能:")
        print("1. 编辑数据库集合（查看、导出、清空、新建集合）")
        print("2. 文本嵌入索引（选择集合存储文本-向量对，非空则清空）")
        print("3. 问答查询（针对选定数据库集合进行问答）")
        print("4. Markdown文件图像处理（调用视觉模型为图片生成描述并附加）")
        print("5. 从文件生成问答对（用于构建评测集）")
        print("6. 退出")
        
        try:
            choice = input("\n请输入选项 (1-5): ").strip()
        except EOFError:
            print("\n退出系统")
            break
        
        if choice == '1':
            print("\n--- 数据库集合管理 ---")
            db_manager = DatabaseManager()
            
            while True:
                print("\n数据库管理选项:")
                print("1. 查看现有集合")
                print("2. 集合内容导出（文本-向量对前五维度导出为JSON）")
                print("3. 清空集合（仅删除数据，保留集合）")
                print("4. 创建空集合")
                print("5. 返回主菜单")
                
                sub_choice = input("请选择 (1-5): ").strip()
                
                if sub_choice == '1':
                    db_manager.list_collections()
                elif sub_choice == '2':
                    collection_names = db_manager.list_collections()
                    if collection_names:
                        try:
                            idx = int(input(f"请选择要导出的集合 (1-{len(collection_names)}): ")) - 1
                            if 0 <= idx < len(collection_names):
                                export_collection_content(db_manager, collection_names[idx])
                            else:
                                print("无效的索引")
                        except ValueError:
                            print("请输入有效的数字")
                elif sub_choice == '3':
                    collection_names = db_manager.list_collections()
                    if collection_names:
                        try:
                            idx = int(input(f"请选择要清空的集合 (1-{len(collection_names)}): ")) - 1
                            if 0 <= idx < len(collection_names):
                                db_manager.clear_collection(collection_names[idx])
                            else:
                                print("无效的索引")
                        except ValueError:
                            print("请输入有效的数字")
                elif sub_choice == '4':
                    new_name = input("请输入新集合的名称: ").strip()
                    if new_name:
                        db_manager.create_empty_collection(new_name)
                    else:
                        print("集合名称不能为空")
                elif sub_choice == '5':
                    break
                else:
                    print("无效选项，请重试")
        
        elif choice == '2':
            print("\n--- 文本嵌入索引 ---")
            
            txt_files = display_txt_files()
            if not txt_files:
                print("没有找到可处理的文件，跳过索引")
                continue
            
            print(f"\n请输入要处理的文件序号，支持单选或多选")
            print("例如: 输入 '1' 表示选择第1个文件")
            print("例如: 输入 '1,2,4' 表示选择第1、2、4个文件")
            
            selection_str = input("请选择文件序号: ").strip()
            selected_indices = parse_selection(selection_str, len(txt_files))
            
            if not selected_indices:
                print("没有选择有效的文件，跳过索引")
                continue
            
            selected_files = [txt_files[i] for i in selected_indices]
            print(f"\n已选择以下文件: {selected_files}")
            
            try:
                indexer = QwenIndexer()
                
                collection_names = indexer.db_manager.list_collections()
                
                if not collection_names:
                    print("没有现有集合，将创建新集合")
                    new_name = input("请输入新集合的名称: ").strip()
                    if not new_name:
                        new_name = "default_collection"
                    target_collection = new_name
                else:
                    print("\n请选择目标集合:")
                    for i, name in enumerate(collection_names, 1):
                        print(f"  {i}. {name}")
                    print(f"  {len(collection_names) + 1}. 创建新集合")
                    
                    while True:
                        try:
                            coll_choice = int(input(f"\n请选择 (1-{len(collection_names) + 1}): "))
                            if 1 <= coll_choice <= len(collection_names):
                                target_collection = collection_names[coll_choice - 1]
                                
                                collection = indexer.chroma_client.get_collection(target_collection)
                                count = collection.count()
                                if count > 0:
                                    confirm = input(f"集合 '{target_collection}' 有 {count} 个文档，是否清空? (y/N): ")
                                    if confirm.lower() == 'y':
                                        indexer.db_manager.clear_collection(target_collection)
                                
                                break
                            elif coll_choice == len(collection_names) + 1:
                                new_name = input("请输入新集合的名称: ").strip()
                                if new_name:
                                    target_collection = new_name
                                    break
                                else:
                                    print("集合名称不能为空，请重新选择")
                            else:
                                print(f"请输入 1 到 {len(collection_names) + 1} 之间的数字")
                        except ValueError:
                            print("请输入有效的数字")
                
                for filename in selected_files:
                    file_path = f"../data/src_files/{filename}"
                    print(f"\n正在处理文件: {filename}")
                    
                    indexer.index_single_file_to_collection(file_path, target_collection)
                
                print(f"\n所有选定的文件已成功索引到集合 '{target_collection}'！")
                
            except ValueError as e:
                print(f"错误: {e}")
                print("请确保已设置 OPENAI_API_KEY 环境变量")
            except Exception as e:
                print(f"索引过程中出错: {e}")
        
        elif choice == '3':
            print("\n--- 问答查询 ---")
            interactive_query()

        elif choice == '4':
            handle_image_captioning()

        elif choice == '5':
            run_qa_generation()
        
        elif choice == '6':
            print("再见！")
            break
        
        else:
            print("无效选项，请重试")

if __name__ == "__main__":
    main()