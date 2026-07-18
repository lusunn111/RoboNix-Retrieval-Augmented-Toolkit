"""
轨迹指标计算模块
包含三个计算类：
1. CurvatureCalculator - 曲率半径计算（最小二乘圆法）
2. TrajectoryMetricsCalculator - 轨迹位移指标计算
3. CompositeMetricsCalculator - 综合指标计算
"""

import numpy as np
from scipy.optimize import least_squares


class CurvatureCalculator:
    """
    曲率半径计算器
    使用最小二乘法拟合圆来计算轨迹的曲率半径
    """
    
    def __init__(self, window_size=5, curvature_threshold=0.006):
        """
        初始化曲率半径计算器
        
        参数:
            window_size: 滑动窗口大小，用于计算局部曲率
            curvature_threshold: 曲率半径阈值（单位：米）
                                > threshold: 使用检索（纯DB）
                                <= threshold: 使用verify（AR生成）
        """
        self.window_size = window_size
        self.curvature_threshold = curvature_threshold
        self.action_history = []  # 存储action的x,y,z坐标历史
        
    def update_history(self, action):
        """
        更新action历史记录
        
        参数:
            action: numpy array, shape (7,) - [x, y, z, rx, ry, rz, gripper]
        """
        # 只提取前3维的位置信息 (x, y, z)
        position = action[:3]
        self.action_history.append(position)
        
    def clear_history(self):
        """清空历史记录（用于新episode开始时）"""
        self.action_history = []
        
    def compute_radius_least_squares(self, points):
        """
        使用最小二乘法拟合圆，直接返回曲率半径（半径R）
        
        参数:
            points: (N, 3) 的点数组
            
        返回:
            radius: 曲率半径（单位：米），如果无法计算则返回np.nan
        """
        if len(points) < 3:
            return np.nan
        
        # 使用3D点的投影到最佳拟合平面
        center = np.mean(points, axis=0)
        points_centered = points - center
        
        # SVD找到最佳平面
        try:
            _, _, vh = np.linalg.svd(points_centered)
            normal = vh[2, :]
            
            # 投影到平面
            points_2d = points_centered - np.outer(np.dot(points_centered, normal), normal)
            
            # 使用前两个主成分作为2D坐标
            u = vh[0, :]
            v = vh[1, :]
            x = np.dot(points_2d, u)
            y = np.dot(points_2d, v)
            
            # 拟合圆
            def calc_R(xc, yc):
                return np.sqrt((x - xc)**2 + (y - yc)**2)
            
            def f(c):
                Ri = calc_R(*c)
                return Ri - Ri.mean()
            
            center_estimate = np.array([x.mean(), y.mean()])
            result = least_squares(f, center_estimate)
            xc, yc = result.x
            Ri = calc_R(xc, yc)
            R = Ri.mean()
            
            # 直接返回半径R（曲率半径）
            return R if R > 1e-6 else np.nan
        except Exception as e:
            # SVD或拟合失败时返回nan
            return np.nan
    
    def get_current_radius(self):
        """
        计算当前轨迹的曲率半径
        
        返回:
            radius: 当前的曲率半径（单位：米），如果无法计算则返回np.nan
        """
        if len(self.action_history) < 3:
            # 历史记录不足，无法计算曲率
            return np.nan
        
        # 获取窗口内的点
        start_idx = max(0, len(self.action_history) - self.window_size)
        window_points = np.array(self.action_history[start_idx:])
        
        # 计算曲率半径
        radius = self.compute_radius_least_squares(window_points)
        
        return radius
    
    def should_use_retrieval(self):
        """
        根据当前曲率半径决定是否使用检索
        
        返回:
            bool: True表示使用检索（纯DB），False表示使用verify（AR）
        """
        radius = self.get_current_radius()
        
        # 如果无法计算曲率（历史不足或计算失败），默认使用AR
        if np.isnan(radius):
            return False
        
        # 曲率半径大于阈值 -> 使用检索
        # 曲率半径小于等于阈值 -> 使用verify（AR）
        return radius > self.curvature_threshold
    
    def get_history_length(self):
        """返回当前历史记录的长度"""
        return len(self.action_history)
    
    def get_decision_info(self):
        """
        获取决策相关的详细信息（用于日志记录）
        
        返回:
            dict: 包含radius, threshold, decision等信息
        """
        radius = self.get_current_radius()
        use_retrieval = self.should_use_retrieval()
        
        return {
            'radius': radius,
            'threshold': self.curvature_threshold,
            'history_length': len(self.action_history),
            'use_retrieval': use_retrieval,
            'decision': 'Retrieval' if use_retrieval else 'AR'
        }


class TrajectoryMetricsCalculator:
    """
    轨迹位移指标计算器
    计算滑动窗口内，窗口最后一个点与前面所有点的欧式距离之和
    """
    
    def __init__(self, window_size=5):
        """
        初始化轨迹指标计算器
        
        参数:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.action_history = []  # 存储action的x,y,z坐标历史
        
    def update_history(self, action):
        """
        更新action历史记录
        
        参数:
            action: numpy array, shape (7,) - [x, y, z, rx, ry, rz, gripper]
                   或 shape (3,) - [x, y, z]
        """
        # 提取前3维的位置信息 (x, y, z)
        if len(action) >= 3:
            position = action[:3]
        else:
            position = action
        self.action_history.append(position)
        
    def clear_history(self):
        """清空历史记录（用于新episode开始时）"""
        self.action_history = []
        
    def compute_displacement_metric(self, points):
        """
        计算位移指标：窗口最后一个点与前面所有点的欧式距离之和
        
        参数:
            points: (N, 3) 的点数组
            
        返回:
            metric: 位移指标值，如果无法计算则返回np.nan
        """
        if len(points) < 2:
            return np.nan
        
        # 窗口最后一个点（当前点）
        last_point = points[-1]
        
        # 计算最后一个点与前面所有点的欧式距离之和
        total_distance = 0.0
        for i in range(len(points) - 1):  # 不包括最后一个点自己
            dist = np.linalg.norm(last_point - points[i])
            total_distance += dist
        
        return total_distance
    
    def get_current_metric(self):
        """
        计算当前的位移指标
        
        返回:
            metric: 当前的位移指标值，如果无法计算则返回np.nan
        """
        if len(self.action_history) < 2:
            # 历史记录不足，无法计算
            return np.nan
        
        # 获取窗口内的点：从 max(0, i-window_size+1) 到 i（包含i）
        start_idx = max(0, len(self.action_history) - self.window_size)
        window_points = np.array(self.action_history[start_idx:])
        
        # 计算位移指标
        metric = self.compute_displacement_metric(window_points)
        
        return metric
    
    def get_history_length(self):
        """返回当前历史记录的长度"""
        return len(self.action_history)
    
    def get_metric_info(self):
        """
        获取指标相关的详细信息（用于日志记录）
        
        返回:
            dict: 包含metric, history_length等信息
        """
        metric = self.get_current_metric()
        
        return {
            'displacement_metric': metric,
            'history_length': len(self.action_history),
            'window_size': self.window_size
        }


class CompositeMetricsCalculator:
    """
    综合指标计算器
    结合曲率半径指标和位移指标，进行归一化后加权求和
    """
    
    def __init__(self, window_size=5, 
                 displacement_range=(0.000009, 0.120187),
                 radius_range=(0.000001, 0.014615)):
        """
        初始化综合指标计算器
        
        参数:
            window_size: 滑动窗口大小
            displacement_range: 位移指标的归一化范围 (min, max)
            radius_range: 曲率半径指标的归一化范围 (min, max)
        """
        self.window_size = window_size
        self.displacement_range = displacement_range
        self.radius_range = radius_range
        
        # 创建两个子计算器
        self.curvature_calc = CurvatureCalculator(window_size=window_size)
        self.trajectory_calc = TrajectoryMetricsCalculator(window_size=window_size)
        
    def update_history(self, action):
        """
        更新action历史记录（同时更新两个子计算器）
        
        参数:
            action: numpy array, shape (7,) - [x, y, z, rx, ry, rz, gripper]
                   或 shape (3,) - [x, y, z]
        """
        self.curvature_calc.update_history(action)
        self.trajectory_calc.update_history(action)
        
    def clear_history(self):
        """清空历史记录（用于新episode开始时）"""
        self.curvature_calc.clear_history()
        self.trajectory_calc.clear_history()
        
    def normalize_value(self, value, value_range):
        """
        按照指定范围进行归一化
        低于下限为0，高于上限为1
        
        参数:
            value: 待归一化的值
            value_range: (min, max) 归一化范围
            
        返回:
            normalized_value: 归一化后的值 [0, 1]
        """
        if np.isnan(value):
            return np.nan
        
        min_val, max_val = value_range
        
        # 低于下限为0
        if value <= min_val:
            return 0.0
        # 高于上限为1
        elif value >= max_val:
            return 1.0
        # 线性归一化
        else:
            return (value - min_val) / (max_val - min_val)
    
    def compute_composite_metric(self, alpha=0.5):
        """
        计算综合指标
        
        参数:
            alpha: 曲率半径指标的权重 [0, 1]
                  综合指标 = alpha * 曲率半径指标 + (1-alpha) * 位移指标
                  
        返回:
            composite_metric: 综合指标值 [0, 1]，如果无法计算则返回np.nan
        """
        # 获取原始指标
        radius = self.curvature_calc.get_current_radius()
        displacement = self.trajectory_calc.get_current_metric()
        
        # 归一化
        radius_norm = self.normalize_value(radius, self.radius_range)
        displacement_norm = self.normalize_value(displacement, self.displacement_range)
        
        # 如果有任何一个无法计算，返回nan
        if np.isnan(radius_norm) or np.isnan(displacement_norm):
            return np.nan
        
        # 加权求和
        composite = alpha * radius_norm + (1 - alpha) * displacement_norm
        
        return composite
    
    def get_current_metrics(self, alpha=0.5):
        """
        获取当前所有指标（原始值、归一化值、综合值）
        
        参数:
            alpha: 曲率半径指标的权重
            
        返回:
            dict: 包含所有指标信息
        """
        # 原始值
        radius = self.curvature_calc.get_current_radius()
        displacement = self.trajectory_calc.get_current_metric()
        
        # 归一化值
        radius_norm = self.normalize_value(radius, self.radius_range)
        displacement_norm = self.normalize_value(displacement, self.displacement_range)
        
        # 综合指标
        composite = self.compute_composite_metric(alpha)
        
        return {
            'raw': {
                'radius': radius,
                'displacement': displacement
            },
            'normalized': {
                'radius': radius_norm,
                'displacement': displacement_norm
            },
            'composite': composite,
            'alpha': alpha,
            'history_length': self.curvature_calc.get_history_length()
        }
    
    def get_history_length(self):
        """返回当前历史记录的长度"""
        return self.curvature_calc.get_history_length()


class AmbiguityStateController:
    """
    基于Ambiguity指标变化趋势的Verify/NoVerify状态控制器
    
    控制逻辑：
    - ambiguity上升（变大）→ state = verify (需要AR验证)
    - ambiguity下降（变小）→ state = noverify (直接使用DB检索)
    - ambiguity平稳 → state ^= 1 (交替切换)
    
    Ambiguity定义：top-k检索结果中，所有action相对于7维重心的平均欧氏距离
    """
    
    def __init__(self, 
                 history_window: int = 3,
                 rise_threshold: float = 0.01,
                 fall_threshold: float = 0.01,
                 initial_state: int = 0):
        """
        初始化Ambiguity状态控制器
        
        参数:
            history_window: 历史窗口大小，用于计算ambiguity变化趋势
            rise_threshold: 上升阈值，ambiguity增加超过此值认为是上升
            fall_threshold: 下降阈值，ambiguity减少超过此值认为是下降
            initial_state: 初始状态 (0=noverify/DB, 1=verify/AR)
        """
        self.history_window = history_window
        self.rise_threshold = rise_threshold
        self.fall_threshold = fall_threshold
        
        # 状态：0 = noverify (使用DB), 1 = verify (使用AR)
        self.state = initial_state
        
        # ambiguity历史记录
        self.ambiguity_history = []
        
    def compute_ambiguity(self, actions: list) -> float:
        """
        计算top-k actions的ambiguity（歧义性）
        定义：所有action相对于7维重心的平均欧氏距离
        
        参数:
            actions: List of action arrays, each shape (7,)
            
        返回:
            ambiguity: 平均距离（歧义性越大表示检索结果越分散）
        """
        if len(actions) == 0:
            return np.nan
        
        if len(actions) == 1:
            return 0.0
        
        actions_array = np.array(actions)  # shape: (k, 7)
        
        # 计算7维重心
        centroid = np.mean(actions_array, axis=0)  # shape: (7,)
        
        # 计算每个action到重心的欧氏距离
        distances = [np.linalg.norm(a - centroid) for a in actions_array]
        
        # 返回平均距离
        return np.mean(distances)
    
    def update_and_get_state(self, actions: list) -> tuple:
        """
        根据当前检索结果更新ambiguity历史，并返回推荐的状态
        
        参数:
            actions: List of action arrays from top-k retrieval
            
        返回:
            (state, trend, ambiguity): 
                state: 0=noverify(DB), 1=verify(AR)
                trend: 'rising', 'falling', 'stable'
                ambiguity: 当前计算的ambiguity值
        """
        # 计算当前ambiguity
        current_ambiguity = self.compute_ambiguity(actions)
        
        # 如果无法计算，默认使用verify
        if np.isnan(current_ambiguity):
            return 1, 'unknown', np.nan
        
        # 添加到历史
        self.ambiguity_history.append(current_ambiguity)
        
        # 如果历史不足，默认交替
        if len(self.ambiguity_history) < 2:
            self.state ^= 1
            return self.state, 'initial', current_ambiguity
        
        # 计算变化趋势
        # 使用最近几个点的平均变化
        window_size = min(self.history_window, len(self.ambiguity_history) - 1)
        recent_history = self.ambiguity_history[-(window_size + 1):]
        
        # 计算平均变化
        changes = []
        for i in range(1, len(recent_history)):
            changes.append(recent_history[i] - recent_history[i-1])
        avg_change = np.mean(changes)
        
        # 判断趋势
        if avg_change > self.rise_threshold:
            # 上升 → verify
            trend = 'rising'
            self.state = 1  # verify
        elif avg_change < -self.fall_threshold:
            # 下降 → noverify
            trend = 'falling'
            self.state = 0  # noverify
        else:
            # 平稳 → 交替
            trend = 'stable'
            self.state ^= 1
        
        return self.state, trend, current_ambiguity
    
    def get_current_state(self) -> int:
        """获取当前状态"""
        return self.state
    
    def should_verify(self) -> bool:
        """是否需要verify（使用AR）"""
        return self.state == 1
    
    def should_use_db(self) -> bool:
        """是否使用DB检索（noverify）"""
        return self.state == 0
    
    def clear_history(self):
        """清空历史记录（用于新episode开始时）"""
        self.ambiguity_history = []
        self.state = 0  # 重置为noverify
        
    def get_history_length(self) -> int:
        """返回历史记录长度"""
        return len(self.ambiguity_history)
    
    def get_recent_ambiguities(self, n: int = 5) -> list:
        """获取最近n个ambiguity值"""
        return self.ambiguity_history[-n:] if len(self.ambiguity_history) >= n else self.ambiguity_history.copy()
    
    def get_state_info(self) -> dict:
        """
        获取状态的详细信息（用于日志记录）
        """
        if len(self.ambiguity_history) == 0:
            return {
                'state': self.state,
                'state_name': 'verify' if self.state == 1 else 'noverify',
                'current_ambiguity': np.nan,
                'history_length': 0,
                'recent_ambiguities': []
            }
        
        return {
            'state': self.state,
            'state_name': 'verify' if self.state == 1 else 'noverify',
            'current_ambiguity': self.ambiguity_history[-1],
            'history_length': len(self.ambiguity_history),
            'recent_ambiguities': self.get_recent_ambiguities(5)
        }


if __name__ == "__main__":
    # 测试代码
    print("=" * 80)
    print("测试1: CurvatureCalculator")
    print("=" * 80)
    
    # 创建计算器
    calc = CurvatureCalculator(window_size=5, curvature_threshold=0.006)
    
    # 模拟一些轨迹点（直线）
    print("\n1.1 直线轨迹")
    for i in range(10):
        action = np.array([i * 0.01, 0, 0, 0, 0, 0, 0])  # 沿x轴移动
        calc.update_history(action)
        info = calc.get_decision_info()
        print(f"Step {i}: radius={info['radius']:.6f}m, decision={info['decision']}")
    
    # 清空并测试圆形轨迹
    calc.clear_history()
    print("\n1.2 圆形轨迹 (半径0.1m)")
    for i in range(10):
        theta = i * np.pi / 10
        action = np.array([0.1 * np.cos(theta), 0.1 * np.sin(theta), 0, 0, 0, 0, 0])
        calc.update_history(action)
        info = calc.get_decision_info()
        print(f"Step {i}: radius={info['radius']:.6f}m, decision={info['decision']}")
    
    # 测试小半径圆形
    calc.clear_history()
    print("\n1.3 小半径圆形轨迹 (半径0.003m)")
    for i in range(10):
        theta = i * np.pi / 5
        action = np.array([0.003 * np.cos(theta), 0.003 * np.sin(theta), 0, 0, 0, 0, 0])
        calc.update_history(action)
        info = calc.get_decision_info()
        print(f"Step {i}: radius={info['radius']:.6f}m, decision={info['decision']}")
    
    print("\n" + "=" * 80)
    print("测试2: TrajectoryMetricsCalculator")
    print("=" * 80)
    
    traj_calc = TrajectoryMetricsCalculator(window_size=5)
    
    print("\n2.1 直线轨迹的位移指标")
    for i in range(10):
        action = np.array([i * 0.01, 0, 0])
        traj_calc.update_history(action)
        info = traj_calc.get_metric_info()
        print(f"Step {i}: displacement={info['displacement_metric']:.6f}")
    
    traj_calc.clear_history()
    print("\n2.2 圆形轨迹的位移指标 (半径0.05m)")
    for i in range(10):
        theta = i * np.pi / 10
        action = np.array([0.05 * np.cos(theta), 0.05 * np.sin(theta), 0])
        traj_calc.update_history(action)
        info = traj_calc.get_metric_info()
        print(f"Step {i}: displacement={info['displacement_metric']:.6f}")
    
    print("\n" + "=" * 80)
    print("测试3: CompositeMetricsCalculator")
    print("=" * 80)
    
    comp_calc = CompositeMetricsCalculator(window_size=5)
    
    print("\n3.1 直线轨迹的综合指标")
    for i in range(10):
        action = np.array([i * 0.01, 0, 0, 0, 0, 0, 0])
        comp_calc.update_history(action)
        metrics = comp_calc.get_current_metrics(alpha=0.5)
        print(f"Step {i}:")
        print(f"  Raw: radius={metrics['raw']['radius']:.6f}, displacement={metrics['raw']['displacement']:.6f}")
        print(f"  Norm: radius={metrics['normalized']['radius']:.4f}, displacement={metrics['normalized']['displacement']:.4f}")
        print(f"  Composite: {metrics['composite']:.4f}")
    
    comp_calc.clear_history()
    print("\n3.2 圆形轨迹的综合指标 (alpha=0.7)")
    for i in range(10):
        theta = i * np.pi / 10
        action = np.array([0.05 * np.cos(theta), 0.05 * np.sin(theta), 0, 0, 0, 0, 0])
        comp_calc.update_history(action)
        metrics = comp_calc.get_current_metrics(alpha=0.7)
        print(f"Step {i}:")
        print(f"  Raw: radius={metrics['raw']['radius']:.6f}, displacement={metrics['raw']['displacement']:.6f}")
        print(f"  Norm: radius={metrics['normalized']['radius']:.4f}, displacement={metrics['normalized']['displacement']:.4f}")
        print(f"  Composite: {metrics['composite']:.4f}")
    
    print("\n" + "=" * 80)
    print("测试4: AmbiguityStateController")
    print("=" * 80)
    
    amb_ctrl = AmbiguityStateController(
        history_window=3,
        rise_threshold=0.01,
        fall_threshold=0.01,
        initial_state=0
    )
    
    print("\n4.1 测试ambiguity上升趋势 -> verify")
    # 模拟ambiguity逐渐增大的top-k检索结果
    for i in range(5):
        # 构造越来越分散的actions（ambiguity越来越大）
        spread = 0.02 * (i + 1)  # 分散程度逐渐增大
        actions = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([spread, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, spread, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-spread, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, -spread, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        state, trend, ambiguity = amb_ctrl.update_and_get_state(actions)
        state_name = 'verify' if state == 1 else 'noverify'
        print(f"Step {i}: ambiguity={ambiguity:.4f}, trend={trend}, state={state_name}")
    
    amb_ctrl.clear_history()
    print("\n4.2 测试ambiguity下降趋势 -> noverify")
    # 模拟ambiguity逐渐减小的top-k检索结果
    for i in range(5):
        # 构造越来越集中的actions（ambiguity越来越小）
        spread = 0.1 / (i + 1)  # 分散程度逐渐减小
        actions = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([spread, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, spread, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-spread, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, -spread, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        state, trend, ambiguity = amb_ctrl.update_and_get_state(actions)
        state_name = 'verify' if state == 1 else 'noverify'
        print(f"Step {i}: ambiguity={ambiguity:.4f}, trend={trend}, state={state_name}")
    
    amb_ctrl.clear_history()
    print("\n4.3 测试ambiguity平稳 -> 交替")
    # 模拟ambiguity几乎不变的top-k检索结果
    spread = 0.05
    for i in range(8):
        actions = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([spread, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, spread, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([-spread, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, -spread, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        state, trend, ambiguity = amb_ctrl.update_and_get_state(actions)
        state_name = 'verify' if state == 1 else 'noverify'
        print(f"Step {i}: ambiguity={ambiguity:.4f}, trend={trend}, state={state_name}")