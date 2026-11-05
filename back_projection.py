import numpy as np

from log import timed
import data_sources as ds
import preprocessing

@timed
def cal_back_projection(
    color_np:ds.ColorArray,
    depth_np:ds.GeometryArray,
    fx,fy,u0,v0,s,
    )->tuple[ds.ColorArray,ds.GeometryArray]:
    """反投影：根据像素点和深度值计算彩色三维点

    Args:
        color_np:rgb图像(m,n)
        depth_np:深度图像(m,n)
        fx,fy,u0,v0,s:相机内参

    Returns:        
        output_pts:三维点(N,3)      
        output_cols:三维点的rgb值(N,3)  
    """
    row,col=depth_np.shape
    #位置索引
    us=np.arange(row,dtype=ds.GEOMETRY_DTYPE)
    vs=np.arange(col,dtype=ds.GEOMETRY_DTYPE)
    U,V=np.meshgrid(vs,us)

    X=depth_np*(U-u0)/fx
    Y=depth_np*(V-v0)/fy
    Z=depth_np
    points=np.stack([X,Y,Z],axis=-1).reshape(-1,3)
    colors=color_np.reshape(-1,3)

    output_pts,output_cols = preprocessing.remove_invalid_pts(points,colors)

    return output_pts, output_cols
    