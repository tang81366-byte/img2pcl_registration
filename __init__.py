import open3d as o3d
import numpy as np
from pathlib import Path
import h5py
from scipy.spatial import cKDTree
import single_fragment
import single_registration
import chatgpt_single_registration
import single_icp
import time

def test_visual_sphere_pcl(points,colors,pt1,col1,pt2,col2):
    radius = 100  # 根据点云尺度调整
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(pt1)
    sphere.paint_uniform_color(col1) 
    sphere.compute_vertex_normals()

    sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere1.translate(pt2)
    sphere1.paint_uniform_color(col2) 
    sphere1.compute_vertex_normals()

    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32)/255.0))
    o3d.visualization.draw_geometries([sphere,sphere1,pcd])

def test_visual_cloud_line(pts1,pts2):
    n1,n2=len(pts1),len(pts2)
    col1= np.full((n1, 3), [255, 0, 0])
    col2=np.full((n2,3),[0,255,0])

    pts=np.vstack([pts1,pts2])
    cols=np.vstack([col1,col2])

    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector((cols.astype(np.float32)/255.0))

    lines = np.column_stack([np.arange(n1), np.arange(n1) + n1])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (n1, 1)))

    o3d.visualization.draw_geometries([pcd,line_set]) 

def test_visual_cloud_normal(pts,normals):
    n=len(pts)
    col= np.full((n, 3), [245, 245, 245])

    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector((col.astype(np.float32)/255.0))

    pts_extend=pts+100*normals
    pts_z=np.vstack([pts,pts_extend])

    lines = np.column_stack([np.arange(n), np.arange(n) + n])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts_z)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (n, 1)))

    o3d.visualization.draw_geometries([pcd,line_set]) 

def test_visual_cloud(points,colors):
    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32)/255.0))
    o3d.visualization.draw_geometries([pcd])

def test_visual_registration(pts1,pts2,R,t):
    n1,n2=len(pts1),len(pts2)
    col1= np.full((n1, 3), [255, 0, 0])
    col2=np.full((n2,3),[0,255,0])
    t = t.reshape(3, 1)
    tran_pts=(R@pts1.T+t).T

    test_visual_cloud(np.vstack([tran_pts,pts2]),np.vstack([col1,col2]))

def save2file(save_path,rgb_pts):
    rows,cols=rgb_pts.shape

    import os
    mode = 'a' if os.path.exists(save_path) else 'w'
    with open(save_path, mode) as f:
        f.write(f"{rows} {cols}\n")
        np.savetxt(f, rgb_pts, fmt='%.6f', delimiter=' ')

def get_error(R_real,t_real,R,t):
    R_new=R@R_real.T
    t_new=t-t_real
    cos_theta=(R_new[0][0]+R_new[1][1]+R_new[2][2]-1)/2

    theta=np.arccos(np.clip(cos_theta,-1.0,1.0))
    offset=np.linalg.norm(t_new)
    return theta,offset

def evaluate_R_t(pts1,pts2,R,t,dis_threshold):
    #构建kdtree
    tree=cKDTree(pts2)
    trans_pts=pts1@R.T+np.asarray(t).reshape(3)
    distances, indices = tree.query(
            trans_pts,
            k=1,#查询一个点
            p=2,#欧氏距离
            distance_upper_bound=dis_threshold,#距离上限
        )
    valid = np.isfinite(distances)
    errors=np.sqrt(np.mean(distances[valid]**2))

    fit_num=int(valid.sum())/len(pts1)
    return errors,fit_num


        

def create_single_fragment(data_dir,save_name):
    intrinsic=o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    K=intrinsic.intrinsic_matrix
    
    
    color_folder=data_dir / "color"
    depth_folder=data_dir/ "depth"

    curr_time=time.time()
    files = sorted(p for p in color_folder.iterdir() if p.is_file())# 是否需要对文件排序
    file_num=len(files)
    img_size=single_fragment.get_image_size(files[0])

    # 压缩存储，分块提取
    with h5py.File(str(data_dir / save_name), "w") as f:
        dset_pts = f.create_dataset(
            "points",
            shape=(file_num, img_size, 3),
            dtype=np.float32,
            chunks=(1, img_size, 3),        # 每帧一个 chunk
            compression="lzf",            # 或 "gzip", "blosc"
        )
        dset_col=f.create_dataset(
            "colors",
            shape=(file_num, img_size, 3),
            dtype=np.uint8,
            chunks=(1, img_size, 3),        # 每帧一个 chunk
            compression="lzf",            # 或 "gzip", "blosc"
        )
        dset_trans0=f.create_dataset(
            "ori_trans",
            shape=(file_num,4,4),
            dtype=np.float32,
            chunks=(1, 4, 4),        # 每帧一个 chunk
            compression="lzf",            # 或 "gzip", "blosc"
        )


        pre_pts=[]
        pre_fpfh=[]
        for i,file_path in enumerate(files[:2]):
            path_rgb=str(color_folder / file_path.name)
            path_depth=str(depth_folder / file_path.name)
            # 局部点云
            points0,colors0=single_fragment.estimate_single_fragment(path_rgb,path_depth,K[0][0],K[1][1],K[0][1],K[0][2],K[1][2])
            """ for m in range(len(points)-1):
                pt1,pt2=points[m],points[m+1]
                dis=(pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2+(pt1[2]-pt2[2])**2
                if dis>1:
                    col1= np.full((colors.shape[0], 3), [245, 245, 245])
                    
                    test_visual_sphere_pcl(points,col1,pt1,[1,0,0],pt2,[0,1,0]) """
            
            # 去噪

            # 配准使用有效点
            valid = np.isfinite(points0).all(axis=1)
            valid &= points0[:, 2] > 0

            points = points0[valid]
            colors=colors0[valid]
            """ dset_pts[i]=points
            dset_col[i]=colors """

            #降采样
            s=32
            ds_pts=single_registration.voxel_down_sampling(points,s)
            #计算法向和邻域
            neibor_num=min(len(ds_pts),30)
            normals,neibors=single_registration.estimate_normals(ds_pts,neibor_num,2*s)

            # test_visual_cloud_normal(ds_pts,normals)
            
            #计算FPFH
            neibor_num=min(len(ds_pts),100)
            fpfh=single_registration.estimate_fpfh(len(ds_pts),ds_pts,normals,5*s,neibor_num)
            
            if i==0:
                pre_pts.append(ds_pts)
                pre_fpfh.append(fpfh)
                dset_trans0[i]=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            else:
                #计算特征对
                pairs1,pairs2=single_registration.estimate_pairs(pre_pts[0],ds_pts,pre_fpfh[0],fpfh)
                #test_visual_cloud_line(pairs1,pairs2)
                               
                #计算R,t
                R0,t0=single_registration.execute_ransac(pairs1,pairs2,s)
                # test_visual_registration(pre_pts[0],ds_pts,R0,t0)

                #ICP
                R,t=single_icp.point_to_point_icp(R0,t0,pre_pts[0],ds_pts,s)
                #test_visual_registration(pre_pts[0],ds_pts,R,t)

                """R_real=np.array([[0.9952207825,0.0731495135,-0.0646888104],
                              [-0.0701019649,0.9963749464,0.0481903926],
                              [0.0679793744,-0.0434252669,0.9967417319]],dtype=np.float64)
                t_real=np.array([[1.7128454645],[1.4146349002],[-0.3505739308]],dtype=np.float64)
                theta1,off1=get_error(R_real,t_real,R0,t0) """
                theta2,off2=get_error(R0,t0,R,t)

                e1,fit1=evaluate_R_t(pre_pts[0],ds_pts,R0,t0,s)
                e2,fit2=evaluate_R_t(pre_pts[0],ds_pts,R,t,s)
                
                pre_pts[0]=ds_pts
                pre_fpfh[0]=fpfh

            
        f.flush()
            

    print(time.time()-curr_time)
    print("success")

def create_single_fragment_chatgpt(data_dir,save_name):
    intrinsic=o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    K=intrinsic.intrinsic_matrix
    
    
    color_folder=data_dir / "color"
    depth_folder=data_dir/ "depth"

    curr_time=time.time()
    files = sorted(p for p in color_folder.iterdir() if p.is_file())# 是否需要对文件排序
    file_num=len(files)
    img_size=single_fragment.get_image_size(files[0])

    # 压缩存储，分块提取
    with h5py.File(str(data_dir / save_name), "w") as f:
        dset_pts = f.create_dataset(
            "points",
            shape=(file_num, img_size, 3),
            dtype=np.float32,
            chunks=(1, img_size, 3),        # 每帧一个 chunk
            compression="lzf",            # 或 "gzip", "blosc"
        )
        dset_col=f.create_dataset(
            "colors",
            shape=(file_num, img_size, 3),
            dtype=np.uint8,
            chunks=(1, img_size, 3),        # 每帧一个 chunk
            compression="lzf",            # 或 "gzip", "blosc"
        )
        dset_trans0=f.create_dataset(
            "ori_trans",
            shape=(file_num,4,4),
            dtype=np.float32,
            chunks=(1, 4, 4),        # 每帧一个 chunk
            compression="lzf",            # 或 "gzip", "blosc"
        )


        pre_pts=[]
        pre_fpfh=[]
        for i,file_path in enumerate(files[:2]):
            path_rgb=str(color_folder / file_path.name)
            path_depth=str(depth_folder / file_path.name)
            # 局部点云
            points,colors=single_fragment.estimate_single_fragment(path_rgb,path_depth,K[0][0],K[1][1],K[0][1],K[0][2],K[1][2])
            if i==0:
                pre_pts.append(points)
            else:
                points1=pre_pts[0]
                pre_pts[0]=points

                # points1、points2 分别来自相邻两帧的 estimate_single_fragment()。
                # 假设坐标单位为毫米，32 表示 32 mm。
                result = chatgpt_single_registration.register_two_frames(
                    points1,
                    points,
                    voxel_size=32.0,
                    refine=True,
                    visualize=True,
                )

                R = result["R"]
                t = result["t"]
                T = result["T"]

                # 第一帧转换到第二帧坐标系。
                aligned_points1 = points1 @ R.T + t

    
    

if __name__ == "__main__":
    curr_dir=Path(__file__).parent
    data_dir=curr_dir / "data" / "LoungeRGBDImages"
    create_single_fragment(data_dir,"single_fragment.h5")
    #create_single_fragment_chatgpt(data_dir,"single_fragment.h5")
    
