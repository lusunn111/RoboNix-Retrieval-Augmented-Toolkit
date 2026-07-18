"""
直接从本地Qdrant存储检索（零网络延迟）

使用Qdrant的本地存储模式，完全消除网络通信开销。
"""

import time
import numpy as np
import torch
from qdrant_client import QdrantClient
import argparse


def _search_points(client, collection_name: str, query_vector, limit: int = 10):
    """Version-agnostic search"""
    if hasattr(client, "search"):
        try:
            return client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            pass
    
    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return result.points if hasattr(result, "points") else result


def test_direct_local_retrieval(
    qdrant_storage_path: str,
    collection_name: str = "libero_goal_task_93",
    num_queries: int = 100,
    top_k: int = 5
):
    """
    测试直接从本地Qdrant存储检索的速度
    
    Args:
        qdrant_storage_path: Qdrant本地存储路径
        collection_name: 集合名称
        num_queries: 测试查询次数
        top_k: 返回top-k结果
    """
    print("=" * 80)
    print("Qdrant 本地存储直接检索测试（零网络延迟）")
    print("=" * 80)
    print(f"存储路径: {qdrant_storage_path}")
    print(f"集合名称: {collection_name}")
    print(f"测试查询数: {num_queries}")
    print(f"Top-K: {top_k}")
    print("=" * 80 + "\n")
    
    # 连接本地存储
    print("[1/3] 连接本地Qdrant存储...")
    t_connect_start = time.time()
    
    # 使用path参数直接访问本地存储文件
    client = QdrantClient(path=qdrant_storage_path)
    
    t_connect_end = time.time()
    print(f"✓ 连接成功，耗时: {(t_connect_end - t_connect_start)*1000:.2f}ms\n")
    
    # 检查集合
    print("[2/3] 检查集合...")
    try:
        collection_info = client.get_collection(collection_name)
        embedding_dim = collection_info.config.params.vectors.size
        print(f"✓ 集合存在")
        print(f"  - 向量数量: {collection_info.points_count}")
        print(f"  - 向量维度: {embedding_dim}")
        print(f"  - 距离度量: {collection_info.config.params.vectors.distance}\n")
    except Exception as e:
        print(f"✗ 集合不存在: {e}")
        collections = client.get_collections()
        print("\n可用集合:")
        for col in collections.collections:
            try:
                info = client.get_collection(col.name)
                print(f"  - {col.name} ({info.points_count} points)")
            except:
                print(f"  - {col.name}")
        print(f"\n提示: 使用 --collection 参数指定正确的集合名称")
        return
    
    # 生成测试向量
    print("[3/3] 测试检索速度...")
    query_vectors = np.random.randn(num_queries, embedding_dim).astype(np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    
    print("-" * 80)
    retrieval_times = []
    
    for i in range(num_queries):
        query_vector = query_vectors[i].tolist()
        
        # 纯检索计时
        t_start = time.time()
        results = _search_points(client, collection_name, query_vector, top_k)
        t_end = time.time()
        
        retrieval_time = (t_end - t_start) * 1000
        retrieval_times.append(retrieval_time)
        
        if (i + 1) % 10 == 0:
            avg_so_far = np.mean(retrieval_times)
            print(f"  查询 {i+1}/{num_queries}: {retrieval_time:.2f}ms (平均: {avg_so_far:.2f}ms)")
    
    print("-" * 80 + "\n")
    
    # 统计结果
    retrieval_times = np.array(retrieval_times)
    
    print("=" * 80)
    print("本地存储检索速度统计")
    print("=" * 80)
    print(f"平均检索时间: {np.mean(retrieval_times):.3f}ms")
    print(f"中位数检索时间: {np.median(retrieval_times):.3f}ms")
    print(f"标准差: {np.std(retrieval_times):.3f}ms")
    print(f"最小检索时间: {np.min(retrieval_times):.3f}ms")
    print(f"最大检索时间: {np.max(retrieval_times):.3f}ms")
    print(f"P95检索时间: {np.percentile(retrieval_times, 95):.3f}ms")
    print(f"P99检索时间: {np.percentile(retrieval_times, 99):.3f}ms")
    print("=" * 80 + "\n")
    
    # 对比分析
    print("=" * 80)
    print("对比分析")
    print("=" * 80)
    avg_retrieval = np.mean(retrieval_times)
    drafter_time = 13.93
    
    print(f"Drafter模型时间: {drafter_time:.2f}ms")
    print(f"本地存储检索时间: {avg_retrieval:.2f}ms")
    
    if avg_retrieval < drafter_time:
        speedup = drafter_time / avg_retrieval
        print(f"\n✓ 本地存储检索更快 {speedup:.2f}x")
        print(f"\n优势:")
        print(f"  - 零网络延迟")
        print(f"  - 直接文件系统访问")
        print(f"  - 可以嵌入到推理进程中")
    else:
        slowdown = avg_retrieval / drafter_time
        print(f"\n✗ 本地存储检索慢 {slowdown:.2f}x")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试本地Qdrant存储直接检索速度")
    parser.add_argument("--storage_path", type=str, required=True,
                        help="Qdrant本地存储路径")
    parser.add_argument("--collection", type=str, default="libero_goal_task_93",
                        help="集合名称")
    parser.add_argument("--num_queries", type=int, default=100,
                        help="测试查询次数")
    parser.add_argument("--top_k", type=int, default=5,
                        help="返回top-k结果")
    
    args = parser.parse_args()
    
    test_direct_local_retrieval(
        qdrant_storage_path=args.storage_path,
        collection_name=args.collection,
        num_queries=args.num_queries,
        top_k=args.top_k
    )
