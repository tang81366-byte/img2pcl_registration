import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from typing import TypeAlias
from enum import Enum
from pathlib import Path

#数据精度
GEOMETRY_DTYPE:type[np.floating] = np.float32
"""点坐标、距离、体素大小、半径等"""

COMPUTE_DTYPE:type[np.floating] = np.float32
"""旋转、平移、法向、特征等浮点计算"""

COLOR_DTYPE:type[np.unsignedinteger] = np.uint8
"""RGB 颜色，范围[0,255]"""

INDEX_DTYPE:type[np.unsignedinteger] = np.uint64
"""索引"""

ColorArray: TypeAlias = NDArray[np.uint8]
"""颜色矩阵，范围[0,255]"""

GeometryArray: TypeAlias = NDArray[np.float32]
"""点矩阵"""

#阈值
DIS_RADIO_THRESHOLD=COMPUTE_DTYPE(0.9)
"""fpfh特征对:阈值(最近距离/次近距离)"""

DISTANCE_THRESHOLD:float=COMPUTE_DTYPE(16)
"""阈值(距离)"""

#RANSAC
P_RANSAC:float=0.999
"""P(至少有一次抽样,抽到的3组全是正确匹配内点)"""

W0_RANSAC:float=0.05
"""给定的特征对内点对比例初值"""

SAMPLE_NUM_RANSAC:int=3
"""每次抽取的样本数量"""

#ICP
ICP_MAX_ITERATION:int=100

#降采样
DOWN_SAMPLING_S:int=32
"""降采样体素边长"""

#kd-tree查询点、查询距离
NORMAL_SEARCH_NUM:int=30
"""计算法向:邻域点数量"""

NORMAL_SEARCH_RADIUS:float=2*DOWN_SAMPLING_S
"""计算法向:邻域搜索半径"""

FPFH_SEARCH_NUM:int=100
"""计算FPFH:邻域点数量"""

FPFH_SEARCH_RADIUS=5*DOWN_SAMPLING_S
"""计算FPFH:邻域搜索半径"""

#位姿R,t的收敛条件
TRANSLATION_DIS_THRESHOLD=COMPUTE_DTYPE(1e-3)
"""阈值(平移距离)"""

ROTATION_ANGLE_THRESHOLD=COMPUTE_DTYPE(1e-6)
"""阈值(旋转角度)"""

#原始数据
class DATA_MODE(Enum):
    ONE="LoungeRGBDImages" #Open3d提供的房间数据
    TWO="testImages" #飞机壁板数据

def get_data_folder_path(data_mode:DATA_MODE) -> tuple[Path,Path]:
    curr_dir=Path(__file__).parent
    data_dir=curr_dir / "data" / data_mode.value
    color_folder=data_dir / "color"
    depth_folder=data_dir/ "depth"
    return color_folder,depth_folder

def get_carema_in_matrix(data_mode:DATA_MODE) -> NDArray[np.floating]:
    if data_mode == DATA_MODE.ONE:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    elif data_mode == DATA_MODE.TWO:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=1920,height=1200,fx=2427.972163,fy=2428.861606,cx=947.871784,cy=592.184692,)
    else:
        # 以后增加枚举成员但尚未配置内参时，明确报错。
        raise ValueError(f"数据集尚未配置相机内参：{data_mode.name}")
    #K(3,3)
    K= np.array(intrinsic.intrinsic_matrix,dtype=COMPUTE_DTYPE,copy=True,)
    fx=K[0][0]
    fy=K[1][1]
    s=K[0][1]
    u0=K[0][2]
    v0=K[1][2]
    return fx,fy,s,u0,v0





