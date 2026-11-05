import numpy as np
from scipy.spatial import cKDTree
import math
import random

from log import timed
import data_sources as ds

@timed
def estimate_fpfh(points,normals):
    """计算点云的fpfh特征向量
    
    Args:
        points:点云(N,3)
        normals:法向(N,3)

    Returns:
        fpfhs:fpfh特征向量(N,33)
    """
    pt_num = len(points)
    bin_num = 11
    spfh = np.zeros((pt_num, 3 * bin_num), dtype=ds.COMPUTE_DTYPE)

    if pt_num == 0:
        return spfh

    def bin_index(values, low, high):
        """将数值映射到 [0, bin_num - 1] 的直方图索引。"""
        bins = np.floor((values - low) / (high - low) * bin_num)
        return np.clip(bins, 0, bin_num - 1).astype(np.intp)

    # 一次查询所有点的近邻。
    search_radius=ds.FPFH_SEARCH_RADIUS
    search_num=ds.FPFH_SEARCH_NUM

    k = min(pt_num, search_num)
    kd_tree = cKDTree(points)

    distances, indices = kd_tree.query(
        points,
        k=k,
        p=2,
        distance_upper_bound=search_radius,
    )

    # k=1 时也统一为二维数组。
    distances = distances.reshape(pt_num, k).astype(
        ds.GEOMETRY_DTYPE, copy=False
    )
    indices = indices.reshape(pt_num, k)

    # 排除填充值、重合点和查询点自身。
    valid = (indices < pt_num) & np.isfinite(distances)
    valid &= distances > 1e-12
    valid &= indices != np.arange(pt_num)[:, None]

    # 1. 计算每个点的 SPFH。
    for i in range(pt_num):
        ids = indices[i, valid[i]]
        neighbor_distances = distances[i, valid[i]]

        if ids.size == 0:
            continue

        # 当前点指向邻居的单位方向。
        directions = (
            points[ids] - points[i]
        ) / neighbor_distances[:, None]

        u = normals[i]
        v = np.cross(directions, u)
        lengths = np.linalg.norm(v, axis=1)

        # 排除方向与法向平行、无法建立局部坐标系的点对。
        good = lengths > 1e-12
        if not good.any():
            continue

        directions = directions[good]
        v = v[good] / lengths[good, None]
        neighbor_normals = normals[ids[good]]
        w = np.cross(u, v)

        # alpha、phi ∈ [-1, 1]；theta ∈ [-pi, pi]。
        alpha = np.clip(
            np.sum(v * neighbor_normals, axis=1), -1, 1
        )
        phi = np.clip(
            np.sum(directions * u, axis=1), -1, 1
        )
        theta = np.arctan2(
            np.sum(w * neighbor_normals, axis=1),
            np.sum(u * neighbor_normals, axis=1),
        )

        all_bins = np.concatenate((
            bin_index(alpha, -1, 1),
            bin_index(phi, -1, 1) + bin_num,
            bin_index(theta, -np.pi, np.pi) + 2 * bin_num,
        ))

        histogram = np.bincount(
            all_bins, minlength=3 * bin_num
        ).astype(ds.COMPUTE_DTYPE, copy=False)

        spfh[i] = histogram / len(directions)

    # 2. 按距离倒数加权邻居的 SPFH，得到 FPFH。
    fpfhs = spfh.copy()

    for i in range(pt_num):
        ids = indices[i, valid[i]]
        neighbor_distances = distances[i, valid[i]]

        if ids.size == 0:
            continue

        fpfhs[i] += np.mean(
            spfh[ids] / neighbor_distances[:, None],
            axis=0,
        )

    # 3. 每个点的 33 维特征分别进行 L2 归一化。
    norms = np.linalg.norm(fpfhs, axis=1, keepdims=True)

    fpfhs = np.divide(
        fpfhs,
        norms,
        out=np.zeros_like(fpfhs),
        where=norms > 1e-12,
    )

    return fpfhs

@timed
def fpfh_correspondence(points1,fpfh1,points2,fpfh2):
    """fpfh特征向量配对
    
    Args:
        points1:第一帧点云(N,3)
        fpfh1:fpfh特征向量(N,33)
        points2:第二帧点云
        fpfh2:

    Returns:
        left_pairs:第一帧点云中的对子(N,3)
        right_pairs:
    """
    # 每一行对应一个点的 FPFH 特征。
    fpfh1 = np.ascontiguousarray(fpfh1, dtype=ds.COMPUTE_DTYPE)
    fpfh2 = np.ascontiguousarray(fpfh2, dtype=ds.COMPUTE_DTYPE)

    n, m = len(fpfh1), len(fpfh2)

    # 目标特征不足两个时，无法进行最近邻/次近邻比值检查。
    if n == 0 or m < 2:
        return points1[:0].copy(), points2[:0].copy()

    # 1. 正向查询：为每个 fpfh1 查找 fpfh2 中最近的两个特征。
    tree2 = cKDTree(fpfh2)
    distances, indices = tree2.query(fpfh1, k=2, workers=-1, distance_upper_bound=ds.FPFH_SEARCH_RADIUS)

    d1 = distances[:, 0]  # 最近邻欧氏距离
    d2 = distances[:, 1]  # 次近邻欧氏距离
    idx1 = indices[:, 0]  # 最近邻在 fpfh2 中的索引

    # 2. 反向查询：为每个 fpfh2 查找 fpfh1 中最近的一个特征。
    tree1 = cKDTree(fpfh1)
    _, idx2 = tree1.query(fpfh2, k=1, workers=-1, distance_upper_bound=ds.FPFH_SEARCH_RADIUS)

    # 3. 双向最近邻筛选：两个特征必须互为最近邻。
    is_mutual = idx2[idx1] == np.arange(n)

    # 4. 距离比值筛选，并排除次近邻距离过小的情况。
    ratio_ok = (
        (d2**2 > ds.TRANSLATION_DIS_THRESHOLD ** 2)
        & (d1 < ds.DIS_RADIO_THRESHOLD * d2)
    )

    keep = is_mutual & ratio_ok

    left_pairs = points1[keep]
    right_pairs = points2[idx1[keep]]

    return left_pairs, right_pairs

@timed
def estimate_ransac(pairs1,pairs2): #TODO:拆解
    """RANSAC算法
    
    Args:
        pairs1:第一帧点云中的对子(N,3)
        pairs2:

    Returns:
        R:旋转矩阵(3,3)
        t:平移向量(3,)

    Raises:
        RuntimeError: 未找到有效的刚体变换。
    """
    # 转为配置指定的计算精度，不修改输入数组。
    points1 = np.asarray(pairs1, dtype=ds.COMPUTE_DTYPE)
    points2 = np.asarray(pairs2, dtype=ds.COMPUTE_DTYPE)

    feature_num = len(points1)
    sample_num = ds.SAMPLE_NUM_RANSAC

    if feature_num < sample_num:
        raise ValueError("匹配点对不足三个，无法估计刚体变换")

    threshold = ds.DISTANCE_THRESHOLD
    p = ds.P_RANSAC
    w0 = ds.W0_RANSAC

    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("DISTANCE_THRESHOLD 必须为有限正数")
    if not 0 < p < 1:
        raise ValueError("P_RANSAC 必须在 (0, 1) 内")
    if not 0 < w0 <= 1:
        raise ValueError("W0_RANSAC 必须在 (0, 1] 内")

    threshold_sq = threshold * threshold

    def calculate_iterations(inlier_ratio):
        """根据内点比例计算所需抽样次数。"""
        if inlier_ratio >= 1:
            return 1

        return max(
            1,
            math.ceil(
                math.log1p(-p)
                / math.log1p(-(inlier_ratio ** sample_num))
            ),
        )

    def triangle_distances(points):
        """计算三个采样点构成的三条边长。"""
        edges = points[[0, 0, 1]] - points[[1, 2, 2]]
        return np.linalg.norm(edges, axis=1)

    def is_collinear(points, eps=1e-3):
        """检查三个采样点是否重合或近似共线。"""
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]

        length1 = np.linalg.norm(v1)
        length2 = np.linalg.norm(v2)

        if length1 < 1e-12 or length2 < 1e-12:
            return True

        # 单位方向叉积的长度等于夹角正弦。
        sin_angle = np.linalg.norm(
            np.cross(v1 / length1, v2 / length2)
        )
        return sin_angle < eps

    def fit_transform(source, target):
        """通过 SVD 拟合刚体变换；退化时返回 None。"""
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)

        source_centered = source - source_center
        target_centered = target - target_center

        H = source_centered.T @ target_centered

        # 平面点允许秩为 2；共线或重合点不能确定唯一旋转。
        if np.linalg.matrix_rank(H) < 2:
            return None

        U, _, V_T = np.linalg.svd(H)
        V = V_T.T
        R = V @ U.T

        # 修正反射，确保 det(R) = +1。
        if np.linalg.det(R) < 0:
            V[:, -1] *= -1
            R = V @ U.T

        t = target_center - R @ source_center

        return (
            R.astype(ds.COMPUTE_DTYPE, copy=False),
            t.astype(ds.COMPUTE_DTYPE, copy=False).reshape(3),
        )

    def evaluate_transform(R, t):
        """在全部匹配点对上评估内点数量和内点平方误差和。"""
        residuals = points1 @ R.T + t - points2
        squared_errors = np.sum(residuals * residuals, axis=1)

        mask = squared_errors < threshold_sq
        count = int(np.count_nonzero(mask))
        error = np.sum(squared_errors[mask])

        return mask, count, error

    best_R = None
    best_t = None
    best_mask = np.zeros(feature_num, dtype=bool)
    best_num = 0
    best_error = np.inf

    execute_count = 0
    required_count = calculate_iterations(w0)

    # 1. 随机抽样，寻找最佳模型。
    while execute_count < required_count:
        execute_count += 1

        sample_indices = random.sample(
            range(feature_num), sample_num
        )
        sample1 = points1[sample_indices]
        sample2 = points2[sample_indices]

        if is_collinear(sample1) or is_collinear(sample2):
            continue

        # 刚体变换应保持三角形边长不变。
        edge_difference = np.abs(
            triangle_distances(sample1)
            - triangle_distances(sample2)
        )
        if np.any(edge_difference > threshold):
            continue

        result = fit_transform(sample1, sample2)
        if result is None:
            continue

        R, t = result
        mask, count, error = evaluate_transform(R, t)

        if count < sample_num:
            continue

        improved = (
            count > best_num
            or (count == best_num and error < best_error)
        )
        if not improved:
            continue

        best_R = R
        best_t = t
        best_mask = mask
        best_num = count
        best_error = error

        # 根据当前最佳模型的内点比例更新抽样次数。
        required_count = calculate_iterations(
            best_num / feature_num
        )

    if best_R is None:
        raise RuntimeError(
            "RANSAC 未找到有效变换，请检查匹配点、退化情况和距离阈值"
        )

    # 2. 使用最佳模型的全部内点重新拟合，最多更新 10 次。
    for _ in range(10):
        result = fit_transform(
            points1[best_mask],
            points2[best_mask],
        )
        if result is None:
            break

        R_refined, t_refined = result
        mask, count, error = evaluate_transform(
            R_refined, t_refined
        )

        improved = (
            count > best_num
            or (count == best_num and error < best_error)
        )
        if not improved:
            break

        unchanged = np.array_equal(mask, best_mask)

        # 拟合结果是第一帧到第二帧的完整变换，直接替换。
        best_R = R_refined
        best_t = t_refined
        best_mask = mask
        best_num = count
        best_error = error

        if unchanged:
            break

    return best_R, best_t