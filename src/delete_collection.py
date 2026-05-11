#!/usr/bin/env python3
"""
彻底删除 ChromaDB 集合
用法（在 scripts 目录下运行）：
    cd /path/to/Civil-RAG-LocalHost/scripts
    python delete_collection.py <集合名称>
"""

import os
import sys
import shutil
import sqlite3

# 获取当前脚本所在目录（scripts/）
script_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（scripts/ 的上级目录）
project_root = os.path.dirname(script_dir)

# 添加 src 到 sys.path，以便导入 DatabaseManager
sys.path.insert(0, os.path.join(project_root, 'src'))

# 定义 vectorstore 路径（相对于项目根目录）
VECTORSTORE_PATH = os.path.join(project_root, 'data', 'vectorstore')

# 现在可以导入项目中的 DatabaseManager
try:
    from database_manager import DatabaseManager
except ImportError:
    print("错误：无法导入 DatabaseManager，请确保 scripts/delete_collection.py 位于项目根目录的 scripts 文件夹下")
    sys.exit(1)

def get_physical_folders(collection_name: str) -> list:
    """通过 ChromaDB 元数据库查询集合对应的所有物理文件夹名（segments.id）"""
    chroma_db = os.path.join(VECTORSTORE_PATH, "chroma.sqlite3")
    if not os.path.exists(chroma_db):
        print(f"错误：元数据库不存在 {chroma_db}")
        return []

    conn = sqlite3.connect(chroma_db)
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id
        FROM segments s
        JOIN collections c ON s.collection = c.id
        WHERE c.name = ?
    """, (collection_name,))
    rows = cur.fetchall()
    conn.close()
    return [row[0] for row in rows] if rows else []

def fully_delete_collection(collection_name: str) -> bool:
    # 1. 获取所有物理文件夹 UUID
    folder_ids = get_physical_folders(collection_name)
    if not folder_ids:
        print(f"错误：找不到集合 '{collection_name}' 的任何物理文件夹记录。")
        print("可能原因：集合不存在，或元数据已损坏。")
        return False

    # 2. 删除所有物理文件夹
    deleted_dirs = []
    for fid in folder_ids:
        folder_path = os.path.join(VECTORSTORE_PATH, fid)
        if os.path.exists(folder_path):
            print(f"删除物理文件夹: {folder_path}")
            shutil.rmtree(folder_path)
            deleted_dirs.append(fid)
        else:
            print(f"警告：物理文件夹不存在: {folder_path}")

    if deleted_dirs:
        print(f"  ✓ 已删除 {len(deleted_dirs)} 个物理文件夹")

    # 3. 调用 DatabaseManager 删除元数据
    db_manager = DatabaseManager()
    try:
        # 假设 DatabaseManager 有 delete_collection 方法
        db_manager.delete_collection(collection_name)
        print("  ✓ 元数据已通过 API 删除")
    except AttributeError:
        # 可能方法名不同，尝试调用 client.delete_collection
        try:
            db_manager.client.delete_collection(collection_name)
            print("  ✓ 元数据已通过 client.delete_collection 删除")
        except Exception as e2:
            print(f"  ✗ API 删除元数据失败: {e2}")
            # 降级：直接操作 SQLite
            chroma_db = os.path.join(VECTORSTORE_PATH, "chroma.sqlite3")
            if os.path.exists(chroma_db):
                conn = sqlite3.connect(chroma_db)
                cur = conn.cursor()
                cur.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
                row = cur.fetchone()
                if row:
                    coll_id = row[0]
                    cur.execute("DELETE FROM segments WHERE collection = ?", (coll_id,))
                    cur.execute("DELETE FROM collections WHERE id = ?", (coll_id,))
                    conn.commit()
                    print("  已通过 SQL 强制删除 collections 和 segments 记录")
                else:
                    print("  未找到集合的元数据记录，可能已被删除")
                conn.close()
    except Exception as e:
        print(f"  ✗ API 删除元数据失败: {e}")
        # 降级：直接操作 SQLite
        chroma_db = os.path.join(VECTORSTORE_PATH, "chroma.sqlite3")
        if os.path.exists(chroma_db):
            conn = sqlite3.connect(chroma_db)
            cur = conn.cursor()
            cur.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
            row = cur.fetchone()
            if row:
                coll_id = row[0]
                cur.execute("DELETE FROM segments WHERE collection = ?", (coll_id,))
                cur.execute("DELETE FROM collections WHERE id = ?", (coll_id,))
                conn.commit()
                print("  已通过 SQL 强制删除 collections 和 segments 记录")
            else:
                print("  未找到集合的元数据记录，可能已被删除")
            conn.close()

    print(f"✅ 集合 '{collection_name}' 已彻底清理")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python delete_collection.py <集合名称>")
        sys.exit(1)
    fully_delete_collection(sys.argv[1])