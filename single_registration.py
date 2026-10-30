#2.粗配准
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import math



# 体素降采样：
# 输入：点云，体素边长
# 计算点云包围盒
# 计算每个点在包围盒中的体素位置
# 计算不为空的体素格的代表点：质点
def voxel_down_sampling(points,s):
    x0,y0,z0=points.min(axis=0)
    x_max,y_max,z_max=points.max(axis=0)

    x_size=int((x_max-x0)/s)+1
    y_size=int((y_max-y0)/s)+1

    def get_code(indexs,l,w): return indexs@np.array([1,w,l*w],dtype=np.uint64)

    voxel_indexs=np.floor((points-np.array([x0,y0,z0]))/s).astype(np.uint64)
    voxel_codes=get_code(voxel_indexs,y_size,x_size)

    order=np.argsort(voxel_codes,kind = "stable")
    sort_codes=voxel_codes[order]
    sort_points=points[order]
    keys,start_idx=np.unique(sort_codes,return_index=True)
    starts=list(start_idx)+[len(sort_codes)]

    out_nums=len(keys)
    out_pts=np.zeros((out_nums,3),dtype=np.float32)
    for i,key in enumerate(keys):
        out_pts[i]=sort_points[starts[i]:starts[i+1]].mean(axis=0)

    return out_pts
        
    
# 计算点云法向
# 1.为点集建立空间索引：KD树
# 2.搜索近邻点：K近邻 or
# 3.计算协方差矩阵：PCA
# 4.法线：协方差矩阵的最小特征值对应的特征向量
# 5.解决法线朝向问题：统一指向相机坐标系原点
# 6.归一化
def estimate_normals(points,neibor_num,radius):
    pt_num=len(points)
    if pt_num > 0:
        kd_tree=cKDTree(points) # TODO:建树前要进行数据清理，剔除无效点
        normals = np.zeros((pt_num,3),dtype=np.float64)
        #min_eigenvalues=np.zeros((points.shape[0],1),dtype=np.float64)
        neibor_indexs=[]
        
        k=min(len(points),neibor_num)
        for i,pt in enumerate(points):
            idxs=knn_search(pt,kd_tree,radius,k)
            neighbors=points[idxs]
            neighbors_dece=pts_decentralize(neighbors)
            neighbors_dece_T=neighbors_dece.T
            C=neighbors_dece_T @ neighbors_dece / neighbors_dece.shape[0]
            eigvals,eigvecs=np.linalg.eigh(C)
            normals[i]=check_normal_direction(pt,eigvecs[:,0])
            #min_eigenvalues[i]=eigvals[0]
            neibor_indexs.append(idxs)
        # TODO:邻域退化问题、法线的朝向问题待考虑

        return normals,neibor_indexs
    else:
        return np.empty((0,3)),np.empty((0,neibor_num))

# 计算FPFH
# 输入：点云总量，点，法向，点的邻域点
# 1.计算SPFH
# 2.邻域距离加权
def estimate_fpfh(pt_num,points,normals,neibor_radus,neibor_num=100):
    spfh = np.zeros((pt_num,33),dtype=np.float32)
    fpfh=np.copy(spfh)
    devide_num=11

    # alpha、phi ∈ [-1, 1]
    # theta ∈ [-pi, pi]
    def bin_index(value, low, high, bins=11):
        id = int(np.floor((value - low) / (high - low) * bins))
        return int(np.clip(id, 0, bins - 1))

    fpfh_num=min(pt_num,neibor_num)
    neibors=[]
    kd_tree=cKDTree(points)
    for i in range(pt_num):
        pt=points[i]
        normal=normals[i]
        valid_count=0

        
        
        neibors_fpfh_indexs=knn_search(pt,kd_tree,neibor_radus,fpfh_num,True)
        neibors.append(neibors_fpfh_indexs)
        for index in neibors_fpfh_indexs:
            if index == i:
                continue
            neibor_pt=points[index]
            neibor_normal=normals[index]

            """theta1=cal_angle(normal,neibor_normal)
            theta2=cal_angle(normal,neibor_pt-pt)
            theta3=cal_angle(np.cross(normal,neibor_pt-pt),neibor_normal)

            ti1=min(int(theta1/delta),devide_num-1)
            ti2=min(int(theta2/delta),devide_num-1)+ devide_num
            ti3=min(int(theta3/delta%devide_num),devide_num-1) + devide_num*2"""

            d=neibor_pt-pt
            distance=np.linalg.norm(d)
            if distance<=1e-12:#共点
                continue
            d=d/distance
            u=normal
            v=np.cross(d,u)
            v_length=np.linalg.norm(v)
            if v_length<=1e-12:#共线
                continue
            v=v/v_length
            w=np.cross(u,v)

            alpha = np.clip(np.dot(v, neibor_normal), -1.0, 1.0)
            phi = np.clip(np.dot(u, d), -1.0, 1.0)
            theta = np.arctan2(
                np.dot(w, neibor_normal),
                np.dot(u, neibor_normal),
            )

            ti1 = bin_index(alpha, -1, 1)
            ti2 = bin_index(phi, -1, 1) + devide_num
            ti3 = bin_index(theta, -np.pi, np.pi) + devide_num*2

            spfh[i][ti1]+=1
            spfh[i][ti2]+=1
            spfh[i][ti3]+=1
            valid_count+=1

        # 归一化
        if valid_count>0:
            spfh[i]/=valid_count


    for i in range(pt_num):
        pt=points[i]
        neibors_index=np.asarray(neibors[i],dtype=np.uint32)
        neibors_index=neibors_index[neibors_index!=i]

        distances=np.linalg.norm(points[neibors_index]-pt,axis=1)
        valid_ds=distances>1e-12
        neibors_index=neibors_index[valid_ds]
        distances=distances[valid_ds]
        if len(neibors_index)==0:
            continue

        weight=1.0/distances
        fpfh[i] = spfh[i] + (weight[:, None] * spfh[neibors_index]).mean(axis=0)
    return fpfh

# 计算特征点对：
# 1.计算相邻两帧点云的欧氏距离 return:n*m矩阵
# (p-q)^2=p^2-2pq+q^2
# 2.最有匹配点对：每行的最小值同时也是每列的最小值
def estimate_pairs(points1, points2, fpfh1,fpfh2):
    n=len(fpfh1)
    m=len(fpfh2)

    pp_mid = np.sum(fpfh1 * fpfh1, axis=1)  # shape (n,)
    pp = np.repeat(pp_mid[:, None], m, axis=1)  # (n,m)

    qq_mid=np.sum(fpfh2*fpfh2,axis=1)
    qq=np.repeat(qq_mid[None,:],n,axis=0)

    pq=fpfh1@(fpfh2.T)

    distance=pp-2*pq+qq

    # 浮点运算可能产生极小负值
    np.maximum(distance, 0.0, out=distance)

    # TODO：判断最近和次近是否明显

    idx1 = np.argmin(distance, axis=1)
    idx2 = np.argmin(distance, axis=0)

    output_pts1=[]
    output_pts2=[]
    for i in range(n):
        if i==idx2[idx1[i]]:
            output_pts1.append(points1[i])
            output_pts2.append(points2[idx1[i]])

    return np.array(output_pts1),np.array(output_pts2)

# RANSAC剔除无匹配
#收敛次数：
#w每次选中内点的概率
#s每次选的样本数
#p希望至少一次全选中内点的概率
#每轮计算后更新w=当前模型下的内点数/总点数
# 1.跳过不满足刚体变换距离一致性的点对
def execute_ransac(points1,points2,threshold, p=0.999, max_execute_num=1000):
    feature_num=len(points1)
    best_R=np.eye(3,dtype=np.float32)
    best_t=np.zeros((3,1),dtype=np.float32)
    s=3
    if feature_num>=s:
        def cal_dis(pts):
            pts = np.asarray(pts)
            d01 = math.sqrt(np.sum((pts[0] - pts[1]) ** 2))
            d02 = math.sqrt(np.sum((pts[0] - pts[2]) ** 2))
            d12 = math.sqrt(np.sum((pts[1] - pts[2]) ** 2))
            return np.array([d01, d02, d12])    
        def is_collinear(pts, eps=1e-3):
            pts = np.asarray(pts, dtype=float)
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-12 or n2 < 1e-12:
                return True   # 有重合点，视为退化
            if pts.shape[1] == 2:
                cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
            else:
                cross = np.linalg.norm(np.cross(v1, v2))
            return cross / (n1 * n2) < eps
        def cal_N(w):
            N=max_execute_num#分母不为0
            if w>=1:
                N=1
            elif w<=0:
                pass
            else:
                N_denominator=math.log(1-w**s)
                N=math.ceil(math.log(1-p)/N_denominator)
            return N

        import random
        execute_count=0

        
        best_num=0
        best_error=0

        N=max_execute_num
        while execute_count<min(N,max_execute_num):
            execute_count+=1
            nos = random.sample(range(feature_num), 3)    
            pts1 = points1[nos]
            pts2 = points2[nos]
            #共线判断
            if is_collinear(pts1) or is_collinear(pts2):
                continue
            #刚体一致性检查
            delta_dist = np.abs(cal_dis(pts1) - cal_dis(pts2)) 

            if np.any(delta_dist>threshold):
                continue

            #计算初始R,t
            #构造协方差矩阵
            pts1_dece=pts_decentralize(pts1)
            pts2_dece=pts_decentralize(pts2)
            H=pts1_dece.T @ pts2_dece
            U,S,V_T=np.linalg.svd(H)
            V=V_T.T

            R=V@U.T

            # 修正反射(旋转+镜像)，使|R|=1
            if np.linalg.det(R)<0:
                V[:,-1]*=-1
                R=V@U.T

            t=pts2.mean(axis=0)-R @ pts1.mean(axis=0)

            #计算误差和内点数
            residual = points1 @ R.T + t - points2       # (K, 3)
            error_sq = np.sum(residual ** 2, axis=1)     # (K,)

            mask = error_sq < threshold ** 2
            inpt_num = np.count_nonzero(mask)
            error = np.sum(error_sq[mask])

            if N==0 or inpt_num>best_num or (inpt_num == best_num and error < best_error):
                best_R=R
                best_t=t
                best_num=inpt_num
                best_error=error
                w=best_num/feature_num
                N=cal_N(w)
            
    return best_R,best_t

# 评估粗配准结果
def evaluate_result():
    pass
            
            


    

def knn_search(point, kd_tree: cKDTree,radius, k=30,need_radius=True):
    """
    对单个点做 KNN 搜索并加半径约束。
    返回 (idx_filtered, dist_filtered)，均为 1D ndarray。
    """
    point = np.asarray(point, dtype=np.float64).reshape(3,)
    n = kd_tree.n

    if need_radius:
        # distance_upper_bound 即半径约束：超出半径的邻居距离为 inf，idx 为 n
        dists, idxs = kd_tree.query(point, k, distance_upper_bound=radius)
        valid = idxs < n
        idx_filtered = idxs[valid]
    else:
        # 不做半径约束，直接取最近 k 个
        dists, idxs = kd_tree.query(point, k)
        idx_filtered = np.asarray(idxs, dtype=np.intp).reshape(-1)
    return idx_filtered

def pts_decentralize(points):
    pt_mean=points.mean(axis=0)
    pts_dece=points-pt_mean
    return pts_dece

#方向指向光心，归一化
def check_normal_direction(pt,normal):
    camera_pt=np.array([0,0,0])
    norm=np.linalg.norm(normal)
    if (camera_pt-pt)@normal>0:
        return normal/norm
    else:
        return -normal/norm

# 计算夹角
def cal_angle(vec1,vec2):
    cos_theta=(vec1@vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))   # 弧度
    return theta


    



