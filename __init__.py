import data_sources as ds
import preprocessing
import back_projection
import coarse_registration
import fine_registration
import common_op

def main():
    data_mode=ds.DATA_MODE.ONE
    color_folder,depth_folder=ds.get_data_folder_path(data_mode)
    fx,fy,s,u0,v0=ds.get_carema_in_matrix(data_mode)

    files = sorted(p for p in color_folder.iterdir() if p.is_file())# 是否需要对文件排序
    files_depth = sorted(p for p in depth_folder.iterdir() if p.is_file())# 是否需要对文件排序
    file_num=len(files)

    pre_pts=None
    pre_cols=None
    pre_fpfh=None
    for i, (path_rgb, path_depth) in enumerate(zip(files[:2], files_depth[:2])):
        color_np, depth_np=common_op.read_img_from_path(path_rgb,path_depth)
        
        origin_points,origin_colors=back_projection.cal_back_projection(color_np,depth_np,fx,fy,u0,v0,s)
        
        points0,colors0=preprocessing.voxel_down_sampling(origin_points,origin_colors)
        normals,points,colors=preprocessing.estimate_normals(points0,colors0)
        fpfhs=coarse_registration.estimate_fpfh(points,normals)
        if i==0:
            pre_pts=points;pre_cols=colors;pre_fpfh=fpfhs
        else:
            pairs1,pairs2=coarse_registration.fpfh_correspondence(pre_pts,pre_fpfh,points,fpfhs)
            R0,t0=coarse_registration.estimate_ransac(pairs1,pairs2)

            R,t=fine_registration.estimate_icp_p2p(pre_pts,points,R0,t0)

            common_op.visual_registration(pre_pts,pre_cols,points,colors,R,t)


if __name__ == "__main__":
    main()