#3.精配准   点到点ICP
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def point_to_point_icp(R,t,source_pts,target_pts,threshold,dis_threshold=1e-3,angle_threshold=1e-6,max_count=100):
    #构建kdtree
    tree=cKDTree(target_pts)

    R_total = np.asarray(R, dtype=np.float64).copy()
    t_total = np.asarray(t, dtype=np.float64).reshape(3).copy()

    # 创建显示对象
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts)
    source_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)
    target_pcd.paint_uniform_color([0.0, 1.0, 0.0])

    vis = o3d.visualization.Visualizer()
    if not vis.create_window(
        window_name="ICP: orange=source, blue=target",
        width=1280,
        height=800,
    ):
        raise RuntimeError("无法创建 Open3D 可视化窗口")

    vis.add_geometry(target_pcd)
    vis.add_geometry(source_pcd)

    options = vis.get_render_option()
    options.background_color = np.array([255, 255, 255])
    options.point_size = 4.0

    # 显示初始配准状态
    vis.update_renderer()
    window_open = vis.poll_events()

    for _ in range(max_count):
        trans_pts=source_pts@R_total.T+np.asarray(t_total).reshape(3)
        # 5. 更新显示对象并刷新窗口
        source_pcd.points = o3d.utility.Vector3dVector(trans_pts)
        vis.update_geometry(source_pcd)
        vis.update_renderer()
        window_open = vis.poll_events()

        distances, indices = tree.query(
            trans_pts,
            k=1,#查询一个点
            p=2,#欧氏距离
            distance_upper_bound=threshold,#距离上限
        )
        valid = np.isfinite(distances)
        if valid.sum()<3:
            raise ValueError("有效点对不足三个，无法可靠估计刚体变换")

        #邻近点对
        matched_source = trans_pts[valid]
        matched_target = target_pts[indices[valid]]

        #去中心化
        pts_s_dece=pts_decentralize(matched_source)
        pts_t_dece=pts_decentralize(matched_target)

        #最小二乘求位姿：svd
        H=pts_s_dece.T @ pts_t_dece
        # rank=2 的非共线平面点可以求解，不应要求必须 rank=3
        if np.linalg.matrix_rank(H) < 2:
            raise ValueError("匹配点对退化，无法唯一确定旋转")

        U,S,V_T=np.linalg.svd(H)
        V=V_T.T

        R_new=V@U.T

        # 修正反射(旋转+镜像)，使|R|=1
        if np.linalg.det(R_new)<0:
            V[:,-1]*=-1
            R_new=V@U.T
        t_new=matched_target.mean(axis=0)-R_new @ matched_source.mean(axis=0)

        R_total=R_new@R_total
        t_total=R_new@t_total+t_new

        cos_theta=(R_new[0][0]+R_new[1][1]+R_new[2][2]-1)/2
        theta=np.arccos(np.clip(cos_theta,-1.0,1.0))

        if theta<angle_threshold and np.linalg.norm(t_new)<dis_threshold:
            break

    vis.run()
    return R_total,t_total

def pts_decentralize(points):
    pt_mean=points.mean(axis=0)
    pts_dece=points-pt_mean
    return pts_dece






    
