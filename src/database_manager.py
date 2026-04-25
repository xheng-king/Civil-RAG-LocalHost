import chromadb
import os

class DatabaseManager:
    def __init__(self, persist_directory="../data/vectorstore"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
    
    def list_collections(self):
        try:
            collections = self.client.list_collections()
            print("当前存在的集合:")
            collection_names = []
            for i, coll in enumerate(collections, 1):
                try:
                    count = self.client.get_collection(coll.name).count()
                    print(f"  {i}. {coll.name} (文档数: {count})")
                    collection_names.append(coll.name)
                except Exception as e:
                    print(f"  {i}. {coll.name} (无法获取文档数: {e})")
                    collection_names.append(coll.name)
            return collection_names
        except Exception as e:
            print(f"无法访问数据库: {e}")
            return []
    
    def clear_collection(self, collection_name: str):
        try:
            collection = self.client.get_collection(name=collection_name)
            all_result = collection.get()
            all_ids = all_result['ids']
            
            if all_ids:
                collection.delete(ids=all_ids)
                print(f"集合 '{collection_name}' 已清空")
            else:
                print(f"集合 '{collection_name}' 本身已为空")
            return True
        except Exception as e:
            print(f"清空集合时出错: {e}")
            return False
    
    def create_empty_collection(self, collection_name: str):
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            print(f"集合 '{collection_name}' 已创建或已存在")
            return True
        except Exception as e:
            print(f"创建集合时出错: {e}")
            return False