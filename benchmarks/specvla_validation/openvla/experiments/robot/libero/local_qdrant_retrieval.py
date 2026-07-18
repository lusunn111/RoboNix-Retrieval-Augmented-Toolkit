"""
Qdrant本地检索引擎（可嵌入VLA推理进程）

完全消除网络通信，直接从本地Qdrant存储检索。
"""

import time
import numpy as np
import torch
from qdrant_client import QdrantClient
from typing import Dict, List, Optional, Tuple
import logging


class LocalQdrantRetrieval:
    """
    本地Qdrant检索引擎（零网络延迟）
    
    用法示例:
        retrieval = LocalQdrantRetrieval(
            storage_path="/path/to/qdrant_storage"
        )
        results = retrieval.search(
            embedding=query_vector,
            collection_name="libero_goal_task_93",
            top_k=5
        )
    """
    
    def __init__(
        self,
        storage_path: str,
        preload_collections: Optional[List[str]] = None,
        cache_payloads: bool = True
    ):
        """
        初始化本地检索引擎
        
        Args:
            storage_path: Qdrant本地存储路径
            preload_collections: 预加载的集合列表（可选）
            cache_payloads: 是否缓存payload到内存
        """
        self.storage_path = storage_path
        self.cache_payloads = cache_payloads
        
        # 连接本地存储
        self.client = QdrantClient(path=storage_path)
        
        # Payload缓存
        self.payload_cache: Dict[str, Dict[str, Dict]] = {}
        
        # 统计信息
        self.stats = {
            "total_searches": 0,
            "total_search_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # 预加载集合
        if preload_collections:
            for collection in preload_collections:
                self._load_collection_payloads(collection)
        
        logging.info(f"LocalQdrantRetrieval initialized with storage: {storage_path}")
    
    def _load_collection_payloads(self, collection_name: str):
        """加载集合的所有payload到内存"""
        if not self.cache_payloads:
            return
        
        if collection_name in self.payload_cache:
            return
        
        logging.info(f"Loading payloads for collection: {collection_name}")
        
        try:
            points_dict = {}
            offset = None
            batch_size = 100
            
            while True:
                records, offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not records:
                    break
                
                for record in records:
                    points_dict[str(record.id)] = record.payload
                
                if offset is None:
                    break
            
            self.payload_cache[collection_name] = points_dict
            logging.info(f"Loaded {len(points_dict)} payloads for {collection_name}")
            
        except Exception as e:
            logging.error(f"Failed to load payloads for {collection_name}: {e}")
    
    def _search_points(self, collection_name: str, query_vector: List[float], limit: int):
        """Version-agnostic search"""
        if hasattr(self.client, "search"):
            try:
                return self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                pass
        
        result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return result.points if hasattr(result, "points") else result
    
    def search(
        self,
        embedding: np.ndarray,
        collection_name: str,
        top_k: int = 5,
        return_timing: bool = False
    ) -> Dict:
        """
        执行检索
        
        Args:
            embedding: 查询向量 (numpy array或torch tensor)
            collection_name: 集合名称
            top_k: 返回top-k结果
            return_timing: 是否返回计时信息
            
        Returns:
            检索结果字典
        """
        # 转换为列表
        if isinstance(embedding, torch.Tensor):
            query_vector = embedding.cpu().numpy().tolist()
        elif isinstance(embedding, np.ndarray):
            query_vector = embedding.tolist()
        else:
            query_vector = embedding
        
        # 确保集合payload已加载
        if self.cache_payloads and collection_name not in self.payload_cache:
            self._load_collection_payloads(collection_name)
        
        # 执行搜索
        t_start = time.time()
        
        try:
            results = self._search_points(collection_name, query_vector, top_k)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
        
        t_end = time.time()
        search_time = (t_end - t_start) * 1000  # ms
        
        # 更新统计
        self.stats["total_searches"] += 1
        self.stats["total_search_time"] += search_time
        
        # 处理结果
        processed_results = []
        for result in results:
            point_id = str(result.id)
            
            # 优先从缓存获取payload
            if self.cache_payloads and collection_name in self.payload_cache:
                payload = self.payload_cache[collection_name].get(point_id)
                if payload:
                    self.stats["cache_hits"] += 1
                else:
                    self.stats["cache_misses"] += 1
                    payload = getattr(result, "payload", {})
            else:
                payload = getattr(result, "payload", {})
            
            processed_results.append({
                "id": point_id,
                "score": result.score,
                "payload": payload
            })
        
        output = {
            "success": True,
            "collection_name": collection_name,
            "num_results": len(processed_results),
            "results": processed_results
        }
        
        if return_timing:
            output["search_time_ms"] = search_time
        
        return output
    
    def get_trajectory(
        self,
        embedding: np.ndarray,
        collection_name: str,
        top_k: int = 1
    ) -> Optional[List]:
        """
        获取action轨迹（便捷方法）
        
        Returns:
            [current_action, next_action1, next_action2, ...]
        """
        result = self.search(embedding, collection_name, top_k)
        
        if not result["success"] or not result["results"]:
            return None
        
        top_result = result["results"][0]
        payload = top_result["payload"]
        
        current_action = payload.get("current_action", [])
        next_actions = payload.get("next_actions", [])
        
        return [current_action] + next_actions
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        avg_search_time = (
            self.stats["total_search_time"] / self.stats["total_searches"]
            if self.stats["total_searches"] > 0
            else 0
        )
        
        return {
            **self.stats,
            "avg_search_time_ms": avg_search_time,
            "collections_loaded": len(self.payload_cache),
            "cache_hit_rate": (
                self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
                if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0
                else 0
            )
        }


# 测试代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--collection", type=str, default="libero_goal_task_93")
    parser.add_argument("--num_queries", type=int, default=100)
    args = parser.parse_args()
    
    print("=" * 80)
    print("LocalQdrantRetrieval 性能测试")
    print("=" * 80)
    
    # 初始化引擎
    print("\n[1/3] 初始化检索引擎...")
    retrieval = LocalQdrantRetrieval(
        storage_path=args.storage_path,
        preload_collections=[args.collection],
        cache_payloads=True
    )
    
    # 获取集合信息
    print(f"\n[2/3] 检查集合: {args.collection}")
    try:
        collection_info = retrieval.client.get_collection(args.collection)
        embedding_dim = collection_info.config.params.vectors.size
        print(f"  - 向量数量: {collection_info.points_count}")
        print(f"  - 向量维度: {embedding_dim}")
    except Exception as e:
        print(f"✗ 错误: {e}")
        exit(1)
    
    # 测试检索
    print(f"\n[3/3] 执行 {args.num_queries} 次检索...")
    query_vectors = np.random.randn(args.num_queries, embedding_dim).astype(np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    
    times = []
    for i, vec in enumerate(query_vectors):
        result = retrieval.search(vec, args.collection, top_k=5, return_timing=True)
        if result["success"]:
            times.append(result["search_time_ms"])
        
        if (i + 1) % 10 == 0:
            print(f"  完成 {i+1}/{args.num_queries}")
    
    # 统计结果
    times = np.array(times)
    stats = retrieval.get_stats()
    
    print("\n" + "=" * 80)
    print("性能统计")
    print("=" * 80)
    print(f"平均检索时间: {np.mean(times):.3f}ms")
    print(f"中位数: {np.median(times):.3f}ms")
    print(f"P95: {np.percentile(times, 95):.3f}ms")
    print(f"P99: {np.percentile(times, 99):.3f}ms")
    print(f"Cache命中率: {stats['cache_hit_rate']*100:.1f}%")
    print("=" * 80)
    
    # 对比
    drafter_time = 13.93
    avg_time = np.mean(times)
    print(f"\nDrafter: {drafter_time:.2f}ms")
    print(f"本地检索: {avg_time:.2f}ms")
    if avg_time < drafter_time:
        print(f"✓ 本地检索快 {drafter_time/avg_time:.2f}x")
    else:
        print(f"✗ 本地检索慢 {avg_time/drafter_time:.2f}x")
