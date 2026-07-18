"""
测试 Qdrant 检索速度（使用与retrieval_libero_goal.py相同的逻辑）

直接连接Qdrant服务器，测量纯检索时间（不含网络到embedding server的延迟）。
"""

import time
import numpy as np
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import argparse
from pathlib import Path
import logging


def _search_points(client, collection_name: str, query_vector, limit: int = 10):
    """
    Version-agnostic Qdrant search with fallback to query_points.
    与 retrieval_libero_goal.py 中的实现完全一致。
    
    Returns a list of ScoredPoint-like objects with .id and .score.
    """
    # Try legacy / common API first
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
            logging.warning(f"Legacy search failed, will try query_points: {e}")

    # Fallback: use query_points (newer API)
    try:
        # For single-vector collections, pass raw vector directly as `query`
        result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        # Some versions return object with `.points`, others return list
        if hasattr(result, "points"):
            return result.points
        return result
    except Exception as e:
        logging.error(f"query_points fallback failed: {e}")
        raise


def test_local_retrieval_speed(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "libero_goal_task_0",
    num_queries: int = 100,
    top_k: int = 5,
    embedding_dim: int = 4096
):
    """
    测试 Qdrant 检索速度（使用与retrieval服务器相同的逻辑）
    
    Args:
        qdrant_host: Qdrant服务器地址
        qdrant_port: Qdrant服务器端口
        collection_name: 集合名称 (如: libero_goal_task_0)
        num_queries: 测试查询次数
        top_k: 返回top-k结果
        embedding_dim: 向量维度
    """
    print("=" * 80)
    print("Qdrant 检索速度测试")
    print("=" * 80)
    print(f"Qdrant服务器: {qdrant_host}:{qdrant_port}")
    print(f"集合名称: {collection_name}")
    print(f"测试查询数: {num_queries}")
    print(f"Top-K: {top_k}")
    print(f"向量维度: {embedding_dim}")
    print("=" * 80 + "\n")
    
    # 1. 连接 Qdrant
    print("[1/4] 连接 Qdrant 服务器...")
    t_connect_start = time.time()
    
    client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=60.0)
    
    t_connect_end = time.time()
    print(f"✓ 连接成功，耗时: {(t_connect_end - t_connect_start)*1000:.2f}ms\n")
    
    # 2. 检查集合是否存在
    print("[2/4] 检查集合...")
    try:
        collection_info = client.get_collection(collection_name)
        print(f"✓ 集合存在")
        print(f"  - 向量数量: {collection_info.points_count}")
        print(f"  - 向量维度: {collection_info.config.params.vectors.size}")
        print(f"  - 距离度量: {collection_info.config.params.vectors.distance}\n")
        
        # 更新embedding_dim为实际维度
        embedding_dim = collection_info.config.params.vectors.size
    except Exception as e:
        print(f"✗ 集合不存在: {e}")
        print("\n可用集合:")
        collections = client.get_collections()
        for col in collections.collections:
            print(f"  - {col.name}")
        return
    
    # 3. 生成随机查询向量（模拟真实场景）
    print("[3/4] 生成测试查询向量...")
    query_vectors = np.random.randn(num_queries, embedding_dim).astype(np.float32)
    # 归一化（如果使用cosine距离）
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    print(f"✓ 生成 {num_queries} 个查询向量\n")
    
    # 4. 测试检索速度
    print("[4/4] 测试检索速度...")
    print("-" * 80)
    
    retrieval_times = []
    
    for i in range(num_queries):
        query_vector = query_vectors[i].tolist()
        
        # 计时：纯检索时间
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_start = time.time()
        
        # 使用与retrieval_libero_goal.py相同的search逻辑
        results = _search_points(
            client=client,
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_end = time.time()
        
        retrieval_time = (t_end - t_start) * 1000  # 转换为ms
        retrieval_times.append(retrieval_time)
        
        # 每10次打印一次进度
        if (i + 1) % 10 == 0:
            avg_so_far = np.mean(retrieval_times)
            print(f"  查询 {i+1}/{num_queries}: {retrieval_time:.2f}ms (平均: {avg_so_far:.2f}ms)")
    
    print("-" * 80 + "\n")
    
    # 5. 统计结果
    retrieval_times = np.array(retrieval_times)
    
    print("=" * 80)
    print("检索速度统计结果")
    print("=" * 80)
    print(f"平均检索时间: {np.mean(retrieval_times):.3f}ms")
    print(f"中位数检索时间: {np.median(retrieval_times):.3f}ms")
    print(f"标准差: {np.std(retrieval_times):.3f}ms")
    print(f"最小检索时间: {np.min(retrieval_times):.3f}ms")
    print(f"最大检索时间: {np.max(retrieval_times):.3f}ms")
    print(f"P95检索时间: {np.percentile(retrieval_times, 95):.3f}ms")
    print(f"P99检索时间: {np.percentile(retrieval_times, 99):.3f}ms")
    print("=" * 80 + "\n")
    
    # 6. 对比分析
    print("=" * 80)
    print("对比分析")
    print("=" * 80)
    avg_retrieval = np.mean(retrieval_times)
    drafter_time = 13.93  # ms (从您的测试数据)
    
    print(f"Drafter模型时间: {drafter_time:.2f}ms")
    print(f"Qdrant检索时间: {avg_retrieval:.2f}ms")
    
    if avg_retrieval < drafter_time:
        speedup = drafter_time / avg_retrieval
        print(f"\n✓ Qdrant检索更快 {speedup:.2f}x")
    else:
        slowdown = avg_retrieval / drafter_time
        print(f"\n✗ Qdrant检索慢 {slowdown:.2f}x")
    
    print("\n关键结论:")
    print(f"- Qdrant纯检索时间: {avg_retrieval:.2f}ms")
    print(f"- 如果通过HTTP增加的网络延迟 < {max(0, drafter_time - avg_retrieval):.2f}ms，")
    print(f"  那么检索方案总时间仍能优于Drafter")
    print(f"- 如果网络延迟 > {max(0, drafter_time - avg_retrieval):.2f}ms，")
    print(f"  需要评估检索质量提升能否抵消时间损失")
    print("=" * 80 + "\n")
    
    # 7. 检查一个样例结果
    print("=" * 80)
    print("样例检索结果")
    print("=" * 80)
    query_vector = query_vectors[0].tolist()
    results = _search_points(
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )
    
    for idx, result in enumerate(results):
        print(f"\nTop-{idx+1} (score={result.score:.4f}):")
        if hasattr(result, 'payload') and result.payload:
            for key, value in result.payload.items():
                if key == 'action_tokens':
                    print(f"  {key}: {value[:10]}..." if len(str(value)) > 50 else f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
    print("=" * 80 + "\n")


def test_with_filter(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "libero_goal_task_0",
    num_queries: int = 50,
    task_filter: str = None
):
    """
    测试带任务过滤的检索速度
    """
    print("=" * 80)
    print("带任务过滤的检索速度测试")
    print("=" * 80)
    
    client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=60.0)
    
    # 生成查询向量
    collection_info = client.get_collection(collection_name)
    embedding_dim = collection_info.config.params.vectors.size
    query_vectors = np.random.randn(num_queries, embedding_dim).astype(np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    
    retrieval_times = []
    
    for i in range(num_queries):
        query_vector = query_vectors[i].tolist()
        
        # 构建过滤条件
        filter_condition = None
        if task_filter:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="task_name",
                        match=MatchValue(value=task_filter)
                    )
                ]
            )
        
        t_start = time.time()
        # Note: _search_points不支持filter，这里使用原始search
        try:
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=filter_condition,
                limit=5,
                with_payload=True,
                with_vectors=False
            )
        except:
            # Fallback for newer API
            results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=filter_condition,
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            if hasattr(results, "points"):
                results = results.points
        t_end = time.time()
        
        retrieval_times.append((t_end - t_start) * 1000)
    
    retrieval_times = np.array(retrieval_times)
    
    print(f"任务过滤: {task_filter if task_filter else '无'}")
    print(f"平均检索时间: {np.mean(retrieval_times):.3f}ms")
    print(f"中位数: {np.median(retrieval_times):.3f}ms")
    print(f"标准差: {np.std(retrieval_times):.3f}ms")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试Qdrant检索速度（使用与retrieval_libero_goal.py相同的逻辑）")
    parser.add_argument("--qdrant_host", type=str, default="localhost",
                        help="Qdrant服务器地址")
    parser.add_argument("--qdrant_port", type=int, default=6333,
                        help="Qdrant服务器端口")
    parser.add_argument("--collection", type=str, default="libero_goal_task_0",
                        help="集合名称 (如: libero_goal_task_0)")
    parser.add_argument("--num_queries", type=int, default=100,
                        help="测试查询次数")
    parser.add_argument("--top_k", type=int, default=5,
                        help="返回top-k结果")
    parser.add_argument("--embedding_dim", type=int, default=4096,
                        help="向量维度 (默认4096，会自动从集合获取实际维度)")
    parser.add_argument("--test_filter", action="store_true",
                        help="是否测试带过滤的检索")
    parser.add_argument("--task_filter", type=str, default=None,
                        help="任务过滤条件")
    
    args = parser.parse_args()
    
    # 主测试
    test_local_retrieval_speed(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection,
        num_queries=args.num_queries,
        top_k=args.top_k,
        embedding_dim=args.embedding_dim
    )
    
    # 可选：测试带过滤的检索
    if args.test_filter:
        print("\n" + "=" * 80)
        print("额外测试：带任务过滤的检索")
        print("=" * 80 + "\n")
        test_with_filter(
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            collection_name=args.collection,
            num_queries=50,
            task_filter=args.task_filter
        )
