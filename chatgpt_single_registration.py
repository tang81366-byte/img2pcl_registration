import copy
import numpy as np
import open3d as o3d

reg = o3d.pipelines.registration


def make_point_cloud(points):
    """
    points: (N, 3)，假设来自各自相机坐标系。
    两帧必须使用相同坐标单位。
    """
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points 必须是 (N, 3) 数组")

    # RGB-D 无效深度常产生原点；同时清除 NaN 和 Inf。
    valid = np.isfinite(points).all(axis=1)
    valid &= np.any(points != 0, axis=1)
    points = points[valid]

    if len(points) < 3:
        raise ValueError("有效点数不足")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.ascontiguousarray(points)
    )
    pcd = pcd.remove_duplicated_points()

    if len(pcd.points) < 3:
        raise ValueError("去重后有效点数不足")

    return pcd


def prepare_point_cloud(pcd, voxel_size):
    """降采样、法向估计、FPFH。"""
    down = pcd.voxel_down_sample(voxel_size)

    if len(down.points) < 3:
        raise ValueError("降采样后点数不足，请减小 voxel_size")

    down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=2.0 * voxel_size,
            max_nn=30,
        )
    )

    # 适用于当前数据：每帧仍处于各自相机坐标系，光心是原点。
    down.orient_normals_towards_camera_location(
        np.zeros(3, dtype=np.float64)
    )

    feature = reg.compute_fpfh_feature(
        down,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=5.0 * voxel_size,
            max_nn=100,
        ),
    )

    # Open3D 的 FPFH 数据排列为 (33, N)。
    if not np.isfinite(np.asarray(feature.data)).all():
        raise ValueError("FPFH 包含无效值，请检查点云和邻域")

    return down, feature


def print_result(name, result):
    print(
        f"{name}: "
        f"fitness={result.fitness:.4f}, "
        f"inlier_rmse={result.inlier_rmse:.6f}, "
        f"对应数={len(result.correspondence_set)}"
    )


def register_two_frames(
    points1,
    points2,
    voxel_size,
    refine=True,
    visualize=True,
):
    """
    将第一帧 points1 配准到第二帧 points2。

    返回字典中的：
        T: (4, 4)
        R: (3, 3)
        t: (3,)

    满足：
        aligned_points1 = points1 @ R.T + t

    voxel_size 与 points 使用相同单位。
    """
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size 必须是有限正数")

    source = make_point_cloud(points1)
    target = make_point_cloud(points2)

    source_down, source_fpfh = prepare_point_cloud(
        source, voxel_size
    )
    target_down, target_fpfh = prepare_point_cloud(
        target, voxel_size
    )

    print(
        f"第一帧：{len(source.points)} → {len(source_down.points)}"
    )
    print(
        f"第二帧：{len(target.points)} → {len(target_down.points)}"
    )

    # 1. FPFH 匹配 + RANSAC。
    # Open3D 内部完成特征最近邻查询，无需构造完整 N×M 矩阵。
    ransac_distance = 1.5 * voxel_size

    coarse = reg.registration_ransac_based_on_feature_matching(
        source=source_down,
        target=target_down,
        source_feature=source_fpfh,
        target_feature=target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=ransac_distance,
        estimation_method=reg.TransformationEstimationPointToPoint(
            with_scaling=False
        ),
        ransac_n=3,
        checkers=[
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(
                ransac_distance
            ),
        ],
        criteria=reg.RANSACConvergenceCriteria(
            max_iteration=100000,
            confidence=0.999,
        ),
    )

    print_result("RANSAC", coarse)

    if (
        len(coarse.correspondence_set) < 3
        or not np.isfinite(coarse.transformation).all()
    ):
        raise RuntimeError(
            "RANSAC 未获得足够的几何对应，不能作为有效配准使用"
        )

    T = coarse.transformation.copy()
    fine = None

    # 2. 可选：以 RANSAC 位姿为初值，执行点到面 ICP。
    if refine:
        icp_distance = 0.4 * voxel_size

        fine = reg.registration_icp(
            source=source_down,
            target=target_down,
            max_correspondence_distance=icp_distance,
            init=T,
            estimation_method=reg.TransformationEstimationPointToPlane(),
            criteria=reg.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=1e-6,
                max_iteration=50,
            ),
        )
        print_result("ICP", fine)

        # 在相同点云、相同阈值下评价两个位姿，
        # 避免直接比较不同阈值下的 RANSAC 和 ICP 指标。
        before = reg.evaluate_registration(
            source_down, target_down, icp_distance, T
        )
        after = reg.evaluate_registration(
            source_down,
            target_down,
            icp_distance,
            fine.transformation,
        )

        before_n = len(before.correspondence_set)
        after_n = len(after.correspondence_set)

        improved = (
            after_n > before_n
            or (
                after_n == before_n
                and after.inlier_rmse < before.inlier_rmse
            )
        )

        if (
            after_n >= 3
            and np.isfinite(fine.transformation).all()
            and improved
        ):
            T = fine.transformation.copy()
        else:
            print("ICP 未改善统一评价指标，保留 RANSAC 位姿")

    R = T[:3, :3].copy()
    t = T[:3, 3].copy()

    print("第一帧 → 第二帧的 R：\n", R)
    print("第一帧 → 第二帧的 t：\n", t)

    if visualize:
        # 深拷贝，避免 transform 修改输入点云。
        aligned = copy.deepcopy(source)
        reference = copy.deepcopy(target)

        aligned.transform(T)
        aligned.paint_uniform_color([1.0, 0.2, 0.2])
        reference.paint_uniform_color([0.2, 1.0, 0.2])

        o3d.visualization.draw_geometries(
            [aligned, reference],
            window_name="红色：变换后的第一帧；绿色：第二帧",
        )

    return {
        "T": T,
        "R": R,
        "t": t,
        "ransac": coarse,
        "icp": fine,
    }