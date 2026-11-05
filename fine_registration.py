import numpy as np
from scipy.spatial import cKDTree

from log import timed
import data_sources as ds

@timed
def estimate_icp_p2p(points1,points2,R0,t0):
    """ICP算法
    
    Args:
        points1:第一帧点云(N,3)
        points2:
        R0:初始位姿(3,3)
        t0:(3,)

    Returns:
        R:精配准位姿(3,3)
        t:(3,)

    """
    source_pts = np.asarray(points1, dtype=ds.COMPUTE_DTYPE)
    target_pts = np.asarray(points2, dtype=ds.COMPUTE_DTYPE)

    # 复制初始位姿，避免修改调用方的数据。
    R_total = np.asarray(R0, dtype=ds.COMPUTE_DTYPE).copy()
    t_total = np.asarray(t0, dtype=ds.COMPUTE_DTYPE).reshape(3).copy()

    if R_total.shape != (3, 3):
        raise ValueError("R0 必须是形状为 (3, 3) 的旋转矩阵")
    if not np.isfinite(R_total).all() or not np.isfinite(t_total).all():
        raise ValueError("初始位姿不能包含 NaN 或 Inf")

    threshold = ds.DISTANCE_THRESHOLD
    dis_threshold = ds.TRANSLATION_DIS_THRESHOLD
    angle_threshold = ds.ROTATION_ANGLE_THRESHOLD
    max_count = ds.ICP_MAX_ITERATION

    for name, value in (
        ("DISTANCE_THRESHOLD", threshold),
        ("TRANSLATION_DIS_THRESHOLD", dis_threshold),
        ("ROTATION_ANGLE_THRESHOLD", angle_threshold),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为有限正数")

    # 目标点云保持不变，只需建立一次 KD-tree。
    tree = cKDTree(target_pts)

    for _ in range(max_count):
        # 使用当前完整位姿变换源点云。
        transformed_pts = source_pts @ R_total.T + t_total

        # 一次查询全部源点的最近邻。
        distances, indices = tree.query(
            transformed_pts,
            k=1,
            p=2,
            distance_upper_bound=threshold,
        )

        valid = np.isfinite(distances) & (indices < len(target_pts))
        if np.count_nonzero(valid) < 3:
            raise ValueError("有效点对不足三个，无法可靠估计刚体变换")

        matched_source = transformed_pts[valid]
        matched_target = target_pts[indices[valid]]

        # 去中心化。
        source_center = matched_source.mean(axis=0)
        target_center = matched_target.mean(axis=0)

        source_centered = matched_source - source_center
        target_centered = matched_target - target_center

        # 求解当前已变换点云到目标点云的增量位姿。
        H = source_centered.T @ target_centered

        if np.linalg.matrix_rank(H) < 2:
            raise ValueError("匹配点对退化，无法唯一确定旋转")

        U, _, V_T = np.linalg.svd(H)
        V = V_T.T
        R_delta = V @ U.T

        # 修正反射，保证 det(R_delta) = +1。
        if np.linalg.det(R_delta) < 0:
            V[:, -1] *= -1
            R_delta = V @ U.T

        t_delta = target_center - R_delta @ source_center

        # 增量左乘当前位姿，得到新的完整位姿。
        R_total = R_delta @ R_total
        t_total = R_delta @ t_total + t_delta

        # 用 atan2 计算增量旋转角，改善小角度下的数值稳定性。
        half = ds.COMPUTE_DTYPE(0.5)
        one = ds.COMPUTE_DTYPE(1.0)

        skew_vector = np.array(
            [
                R_delta[2, 1] - R_delta[1, 2],
                R_delta[0, 2] - R_delta[2, 0],
                R_delta[1, 0] - R_delta[0, 1],
            ],
            dtype=ds.COMPUTE_DTYPE,
        )

        sin_theta = half * np.linalg.norm(skew_vector)
        cos_theta = np.clip(
            half * (np.trace(R_delta) - one),
            -one,
            one,
        )
        theta = np.arctan2(sin_theta, cos_theta)

        # 同时满足旋转和平移收敛条件。
        if (
            theta < angle_threshold
            and np.linalg.norm(t_delta) < dis_threshold
        ):
            break

    return (
        R_total.astype(ds.COMPUTE_DTYPE, copy=False),
        t_total.astype(ds.COMPUTE_DTYPE, copy=False).reshape(3),
    )