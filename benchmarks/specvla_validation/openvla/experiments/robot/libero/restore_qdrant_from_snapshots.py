#!/usr/bin/env python3
"""
从snapshots目录恢复Qdrant集合
"""
import os
import sys
import shutil
from pathlib import Path
from qdrant_client import QdrantClient

def restore_collections(
    snapshots_dir: str = "/path/to/rtcache/snapshots",
    storage_path: str = "/path/to/rtcache/data/qdrant",
    collection_pattern: str = "libero_goal_task_"
):
    """从snapshots恢复Qdrant集合"""
    
    snapshots_path = Path(snapshots_dir)
    if not snapshots_path.exists():
        print(f"✗ Snapshots目录不存在: {snapshots_dir}")
        return
    
    # 查找所有匹配的集合
    collections = [d.name for d in snapshots_path.iterdir() 
                   if d.is_dir() and d.name.startswith(collection_pattern)]
    
    if not collections:
        print(f"✗ 未找到匹配的集合: {collection_pattern}*")
        return
    
    print(f"找到 {len(collections)} 个集合待恢复")
    print("=" * 80)
    
    # 连接到Qdrant
    print(f"\n连接到本地Qdrant: {storage_path}")
    client = QdrantClient(path=storage_path)
    
    # 获取现有集合
    existing = {col.name for col in client.get_collections().collections}
    print(f"现有集合数: {len(existing)}")
    
    # 恢复每个集合
    restored = 0
    skipped = 0
    
    for i, collection_name in enumerate(sorted(collections), 1):
        snapshot_path = snapshots_path / collection_name
        
        # 检查快照内容
        snapshot_files = list(snapshot_path.glob("*.snapshot"))
        if not snapshot_files:
            print(f"[{i}/{len(collections)}] ⊗ {collection_name}: 无快照文件")
            continue
        
        if collection_name in existing:
            print(f"[{i}/{len(collections)}] ⊙ {collection_name}: 已存在，跳过")
            skipped += 1
            continue
        
        # 恢复快照
        try:
            snapshot_file = snapshot_files[0]  # 使用第一个快照文件
            
            # Qdrant的恢复需要通过HTTP API或先创建集合再恢复
            # 这里我们直接复制快照文件到collections目录
            target_dir = Path(storage_path) / "collections" / collection_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制快照内容（需要解压快照或直接复制数据文件）
            # 简化方案：通过Qdrant API恢复
            # 但direct local client不支持recover_snapshot，需要HTTP
            
            print(f"[{i}/{len(collections)}] ✓ {collection_name}: 准备恢复（需要HTTP API）")
            restored += 1
            
        except Exception as e:
            print(f"[{i}/{len(collections)}] ✗ {collection_name}: {e}")
    
    print("\n" + "=" * 80)
    print(f"总结: {restored} 个准备恢复, {skipped} 个已存在")
    print("\n提示: Qdrant快照恢复需要使用HTTP API")
    print("请确保Qdrant服务运行在 http://localhost:6333")
    print("\n恢复命令示例:")
    print("curl -X POST 'http://localhost:6333/collections/{collection_name}/snapshots/upload' \\")
    print("  -H 'Content-Type: multipart/form-data' \\")
    print("  -F 'snapshot=@{snapshot_file}'")


def check_qdrant_service():
    """检查Qdrant HTTP服务是否运行"""
    import requests
    try:
        resp = requests.get("http://localhost:6333/", timeout=2)
        if resp.status_code == 200:
            print("✓ Qdrant HTTP服务正在运行 (localhost:6333)")
            return True
    except:
        pass
    
    print("✗ Qdrant HTTP服务未运行")
    print("启动命令: docker run -p 6333:6333 -v /path/to/storage:/qdrant/storage qdrant/qdrant")
    return False


def restore_via_http(
    snapshots_dir: str = "/path/to/rtcache/snapshots",
    qdrant_url: str = "http://localhost:6333",
    collection_pattern: str = "libero_goal_task_",
    limit: int = 10  # 限制恢复数量，避免太慢
):
    """通过HTTP API恢复快照"""
    import requests
    from qdrant_client import QdrantClient
    
    snapshots_path = Path(snapshots_dir)
    collections = [d.name for d in snapshots_path.iterdir() 
                   if d.is_dir() and d.name.startswith(collection_pattern)][:limit]
    
    if not collections:
        print(f"✗ 未找到集合")
        return
    
    print(f"将恢复 {len(collections)} 个集合（限制前{limit}个）")
    print("=" * 80)
    
    # 检查服务
    if not check_qdrant_service():
        return
    
    client = QdrantClient(url=qdrant_url)
    existing = {col.name for col in client.get_collections().collections}
    
    restored = 0
    for i, collection_name in enumerate(sorted(collections), 1):
        if collection_name in existing:
            print(f"[{i}/{len(collections)}] ⊙ {collection_name}: 已存在")
            continue
        
        snapshot_path = snapshots_path / collection_name
        snapshot_files = list(snapshot_path.glob("*.snapshot"))
        
        if not snapshot_files:
            print(f"[{i}/{len(collections)}] ⊗ {collection_name}: 无快照")
            continue
        
        snapshot_file = snapshot_files[0]
        
        try:
            # 使用Qdrant client恢复（如果支持）
            # 或使用HTTP直接上传
            with open(snapshot_file, 'rb') as f:
                files = {'snapshot': f}
                url = f"{qdrant_url}/collections/{collection_name}/snapshots/upload"
                resp = requests.post(url, files=files, timeout=30)
                
                if resp.status_code == 200:
                    print(f"[{i}/{len(collections)}] ✓ {collection_name}: 恢复成功")
                    restored += 1
                else:
                    print(f"[{i}/{len(collections)}] ✗ {collection_name}: {resp.status_code}")
        except Exception as e:
            print(f"[{i}/{len(collections)}] ✗ {collection_name}: {e}")
    
    print("\n" + "=" * 80)
    print(f"成功恢复: {restored}/{len(collections)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="从snapshots恢复Qdrant集合")
    parser.add_argument("--snapshots_dir", default="/path/to/rtcache/snapshots")
    parser.add_argument("--storage_path", default="/path/to/rtcache/data/qdrant")
    parser.add_argument("--qdrant_url", default="http://localhost:6333")
    parser.add_argument("--method", choices=["check", "http"], default="http")
    parser.add_argument("--limit", type=int, default=10, help="恢复集合数量限制")
    
    args = parser.parse_args()
    
    print("Qdrant集合恢复工具")
    print("=" * 80)
    
    if args.method == "check":
        restore_collections(args.snapshots_dir, args.storage_path)
    else:
        restore_via_http(args.snapshots_dir, args.qdrant_url, limit=args.limit)
