import numpy as np
from scipy.spatial import cKDTree

from log import timed
import data_sources as ds

def remove_invalid_pts(points,colors):
    """剔除无效点：坐标无效、颜色无效、重复点

    Args:
        points:初始点云(N,3)
        colors:初始点云颜色(N,3)

    Returns:
        output_pts:
        outpus_colors:
    """
    # 转换坐标精度；超出目标类型范围的值会变成 Inf，随后剔除。
    with np.errstate(over="ignore", invalid="ignore"):
        points = points.astype(ds.GEOMETRY_DTYPE, copy=False)

    # 坐标必须全部有限。
    valid_points = np.isfinite(points).all(axis=1)

    # 在转换为 uint8 之前检查颜色，避免溢出或截断掩盖无效值。
    valid_colors = (
        np.isfinite(colors).all(axis=1)
        & ((colors >= 0) & (colors <= 255)).all(axis=1)
    )

    # 同步筛选点和颜色。
    valid = valid_points & valid_colors
    output_pts = points[valid]
    output_colors = colors[valid]

    # 按坐标去重，保留首次出现的有效记录及原有顺序。
    _, first_indices = np.unique(
        output_pts, axis=0, return_index=True
    )
    first_indices = np.sort(first_indices)

    output_pts = output_pts[first_indices]
    output_colors = output_colors[first_indices].astype(
        ds.COLOR_DTYPE, copy=False
    )

    return output_pts, output_colors

def remove_invalid_normals(normals,points,colors):
    """删除法向为0的法线、点
    
    Args:
        normals:(N,3)
        points:(N,3)
        colors:(N,3)

    Returns:
        outpus_normals
        output_pts
        outpus_colors        
    """
    # 所有分量有限，并且至少一个分量不为 0。
    valid = (
        np.isfinite(normals).all(axis=1)
        & np.any(normals != 0, axis=1)
    )

    output_normals = normals[valid]
    output_pts = points[valid]
    output_colors = colors[valid]

    return output_normals, output_pts, output_colors

def estimate_normals_without_check(points):
    """计算法向(统一朝向相机坐标系原点),不带检查

    Args:
        points:点云(N,3)

    Returns:
        normals:法向(N,3)
    """
    pt_num = len(points)
    normals = np.zeros((pt_num, 3), dtype=ds.GEOMETRY_DTYPE)

    if pt_num == 0:
        return normals

    search_radius=ds.NORMAL_SEARCH_RADIUS
    search_num=ds.NORMAL_SEARCH_NUM

    kd_tree = cKDTree(points)
    k = min(pt_num, search_num)

    # 一次查询所有点的近邻。
    distances, indices = kd_tree.query(
        points,
        k=k,
        distance_upper_bound=search_radius,
    )

    # k=1 时也统一为二维数组。
    distances = distances.reshape(pt_num, k)
    indices = indices.reshape(pt_num, k)

    # 排除半径内邻居不足时产生的填充值。
    valid_neighbors = np.isfinite(distances) & (indices < pt_num)

    # 用相对容差判断邻域是否近似共线或全部重合。
    relative_tol = 10 * np.finfo(points.dtype).eps #TODO：

    for i in range(pt_num):
        idxs = indices[i, valid_neighbors[i]]
        if len(idxs) < 3:
            continue

        neighbors = points[idxs]
        centered = neighbors - neighbors.mean(axis=0)

        # 缩放后计算协方差，降低平方运算溢出或下溢的风险。
        # 统一缩放不改变协方差矩阵的特征向量。
        scale = np.max(np.abs(centered))
        if scale == 0:
            continue

        centered = centered / scale
        covariance = centered.T @ centered / len(neighbors)

        # 特征值按升序排列。
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # 第二个特征值过小，说明邻域近似共线，无法确定唯一法向。
        if eigenvalues[1] <= relative_tol * eigenvalues[2]:
            continue

        normal = eigenvectors[:, 0]

        # 使法向朝向原点；特征向量本身已经归一化。
        if np.dot(normal, -points[i]) < 0:
            normal = -normal

        normals[i] = normal

    return normals

@timed
def estimate_normals(points, colors):
    """计算法向(统一朝向相机坐标系原点),并剔除法向为0的点

    Args:
        points:点云(N,3)
        colors:(N,3)

    Returns:
        normals:法向(N,3)
        points:
        colors:
    """
    normals=estimate_normals_without_check(points)
    return remove_invalid_normals(normals,points,colors)
    

@timed
def voxel_down_sampling(points,colors):
    """体素降采样：
    
    Args:
        points:点云(N,3)
        colors:(N,3)

    Returns:
        output_pts:降采样后的点云(N,3)
        output_cols
    """
    x0,y0,z0=points.min(axis=0)
    x_max,y_max,z_max=points.max(axis=0)
    s=ds.DOWN_SAMPLING_S

    x_size=int((x_max-x0)/s)+1
    y_size=int((y_max-y0)/s)+1

    def get_code(indexs,l,w): return indexs@np.array([1,w,l*w],dtype=ds.INDEX_DTYPE)

    s=ds.DOWN_SAMPLING_S
    voxel_indexs=np.floor((points-np.array([x0,y0,z0]))/s).astype(ds.INDEX_DTYPE)
    voxel_codes=get_code(voxel_indexs,y_size,x_size)

    order=np.argsort(voxel_codes,kind = "stable")
    sort_codes=voxel_codes[order]
    sort_points=points[order]
    sort_colors = colors[order]  # 颜色与点同步排序

    keys,start_idx=np.unique(sort_codes,return_index=True)
    starts=list(start_idx)+[len(sort_codes)]

    out_nums=len(keys)
    out_pts=np.zeros((out_nums,3),dtype=ds.GEOMETRY_DTYPE)
    out_cols = np.zeros((out_nums, 3), dtype=ds.COLOR_DTYPE)
    for i,key in enumerate(keys):
        out_pts[i]=sort_points[starts[i]:starts[i+1]].mean(axis=0)
        # 用浮点数计算平均颜色，再取整并转为配置的颜色类型。
        mean_color = sort_colors[starts[i]:starts[i + 1]].mean(axis=0, dtype=ds.COMPUTE_DTYPE)
        out_cols[i] = np.clip(np.rint(mean_color), 0, 255).astype(ds.COLOR_DTYPE)

    return out_pts,out_cols
