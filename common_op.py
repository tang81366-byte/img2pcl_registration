from PIL import Image
import numpy as np
import open3d as o3d
from pathlib import Path

import data_sources as ds

def read_img_from_path(
        color_path:str|Path,
        depth_path:str|Path,
        )->tuple[ds.ColorArray,ds.GeometryArray]:
    with Image.open(color_path) as color_img:# with...读取文件后自动关闭
        color_np = np.asarray(color_img, dtype=ds.COLOR_DTYPE)

    with Image.open(depth_path) as depth_img:
        depth_np = np.asarray(depth_img, dtype=ds.GEOMETRY_DTYPE)

    return color_np, depth_np

def visual_cloud(points,colors):
    pcd=o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32)/255.0))
    o3d.visualization.draw_geometries([pcd])

def visual_registration_without_colors(pts1,pts2,R,t):
    n1,n2=len(pts1),len(pts2)
    cols1= np.full((n1, 3), [255, 0, 0])
    cols2=np.full((n2,3),[0,255,0])
    t = t.reshape(3, 1)
    tran_pts=(R@pts1.T+t).T

    visual_cloud(np.vstack([tran_pts,pts2]),np.vstack([cols1,cols2]))

def visual_registration(pts1,cols1,pts2,cols2,R,t):
    n1,n2=len(pts1),len(pts2)
    t = t.reshape(3, 1)
    tran_pts=(R@pts1.T+t).T

    visual_cloud(np.vstack([tran_pts,pts2]),np.vstack([cols1,cols2]))