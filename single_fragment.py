#1.反射投影
import numpy as np
import open3d as o3d
from PIL import Image
import time

# 生成局部点云
# input：
#   rgb img
#   depth img
#   相机内参：焦距f，图像主点距光心的距离(u0, v0)，像素大小(dx,dy)
# 反投影：计算每个像素点（左上角）在相机坐标系下的三维点
# return colors为0～255的uint，使用时要除以255
def estimate_single_fragment(color_path,depth_path,fx,fy,s,u0,v0):
    """ K=np.array([[fx,s,u0],[0,fy,v0],[0,0,1]])
    K_inv=np.linalg.inv(K)

    color_img=Image.open(color_path)
    depth_img=Image.open(depth_path)

    col,row=color_img.size
    points=np.zeros((row*col,3),dtype=np.float32)
    colors=np.zeros((row*col,3),dtype=np.uint8)

    for u in range(row):
        for v in range(col):
            r,g,b=color_img.getpixel([v,u])
            depth=depth_img.getpixel([v,u])

            camera_pt=(depth*K_inv@np.array([[u],[v],[1]])).reshape(-1,3)
            points[u*col+v]=camera_pt
            colors[u*col+v]=[r,g,b]"""

    # 矩阵计算
    """ K=np.array([[fx,s,u0],[0,fy,v0],[0,0,1]])
    K_inv=np.linalg.inv(K)

    color_img=Image.open(color_path)
    depth_img=Image.open(depth_path)
    color_np=np.asarray(color_img,dtype=np.float32)
    depth_np=np.asarray(depth_img,dtype=np.float32)

    row,col=depth_np.shape

    z_np=np.tile(depth_np.reshape(1,-1),(3,1))
    us=np.arange(row,dtype=np.float32)
    vs=np.arange(col,dtype=np.float32)
    U,V=np.meshgrid(vs,us)
    W=np.ones_like(U)
    img_Pts_T=np.stack([U,V,W],axis=0).reshape(3,-1)

    points_T=z_np * (K_inv @ img_Pts_T)

    points=points_T.T
    colors=color_np.reshape(-1,3) """


    #标量计算
    color_img=Image.open(color_path)
    depth_img=Image.open(depth_path)

    color_np=np.asarray(color_img,dtype=np.uint8)
    depth_np=np.asarray(depth_img,dtype=np.float32)
    row,col=depth_np.shape

    #位置索引
    us=np.arange(row,dtype=np.float32)
    vs=np.arange(col,dtype=np.float32)
    U,V=np.meshgrid(vs,us)

    X=depth_np*(U-u0)/fx
    Y=depth_np*(V-v0)/fy
    Z=depth_np
    points=np.stack([X,Y,Z],axis=-1).reshape(-1,3)
    colors=color_np.reshape(-1,3)
  
    return points,colors

def get_image_size(path):
    img=Image.open(path)
    w,h=img.size
    return w*h


