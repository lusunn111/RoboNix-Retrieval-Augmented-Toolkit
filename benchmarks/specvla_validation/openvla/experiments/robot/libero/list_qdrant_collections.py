"""
列出 Qdrant 中的所有集合及其详细信息
"""

import argparse
from qdrant_client import QdrantClient


def list_collections(qdrant_path: str = "./qdrant_storage"):
    """列出所有集合"""
    print("=" * 80)
    print("Qdrant 集合列表")
    print("=" * 80)
    print(f"Qdrant路径: {qdrant_path}\n")
    
    try:
        client = QdrantClient(path=qdrant_path)
        collections = client.get_collections()
        
        if not collections.collections:
            print("✗ 未找到任何集合")
            print("\n可能原因:")
            print("  1. Qdrant路径不正确")
            print("  2. 还没有创建任何集合")
            print("  3. 需要先运行检索系统来创建集合")
            return
        
        print(f"找到 {len(collections.collections)} 个集合:\n")
        
        for idx, col in enumerate(collections.collections, 1):
            print(f"{idx}. {col.name}")
            
            # 获取详细信息
            try:
                info = client.get_collection(col.name)
                print(f"   - 向量数量: {info.points_count}")
                print(f"   - 向量维度: {info.config.params.vectors.size}")
                print(f"   - 距离度量: {info.config.params.vectors.distance}")
                
                # 获取一个样例点来查看payload结构
                try:
                    points = client.scroll(
                        collection_name=col.name,
                        limit=1,
                        with_payload=True,
                        with_vectors=False
                    )
                    if points[0]:
                        sample_point = points[0][0]
                        print(f"   - Payload字段: {list(sample_point.payload.keys())}")
                except:
                    pass
                
                print()
            except Exception as e:
                print(f"   - 无法获取详细信息: {e}\n")
        
        print("=" * 80)
        print("使用说明:")
        print("=" * 80)
        print("使用以下命令测试某个集合的检索速度:\n")
        for col in collections.collections:
            print(f"python test_qdrant_local_retrieval_speed.py \\")
            print(f"    --collection {col.name} \\")
            print(f"    --num_queries 100\n")
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        print("\n可能需要检查:")
        print("  1. Qdrant路径是否正确")
        print("  2. 是否有权限访问该路径")
        print("  3. Qdrant数据是否完整")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="列出Qdrant中的所有集合")
    parser.add_argument("--qdrant_path", type=str, default="./qdrant_storage",
                        help="Qdrant本地存储路径")
    
    args = parser.parse_args()
    list_collections(args.qdrant_path)
