import open3d as o3d
import numpy as np


def create_point_cloud(shape='sphere'):
    """生成示例点云数据."""
    if shape == 'sphere':
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        pcd = mesh.sample_points_uniformly(number_of_points=1000)
    elif shape == 'cube':
        mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
        pcd = mesh.sample_points_uniformly(number_of_points=1000)

    return pcd


def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)


def preprocess_pcd(pcd):
    # 此处可添加点云降噪、滤波等预处理步骤
    return pcd


def icp_registration(source, target, threshold=0.02):
    source_down = source.voxel_down_sample(voxel_size=0.02)
    target_down = target.voxel_down_sample(voxel_size=0.02)

    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 使用ICP算法对齐点云
    icp_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return icp_result.transformation


def integrate_tsdf(source_pcd, transform, volume):
    source_pcd.transform(transform)
    volume.integrate(
        o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.02,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8).add_rgbd_image(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                source_pcd, depth, convert_rgb_to_intensity=True)))


# 主函数
if __name__ == "__main__":
    # 参数初始化
    voxel_size = 0.02
    sdf_trunc = 0.04

    # 创建示例点云数据
    source_pcd = create_point_cloud('sphere')
    target_pcd = create_point_cloud('cube')

    # 设定目标点云一定的平移和旋转，以给定初始猜测
    target_transform = np.eye(4)
    target_transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0.0, 0.0, 0.5))
    target_transform[:3, 3] = np.array([0.0, 0.5, 0.0])
    target_pcd.transform(target_transform)

    # 可视化初始点云
    o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="Initial Point Clouds")

    # 点云预处理
    source_pcd = preprocess_pcd(source_pcd)
    target_pcd = preprocess_pcd(target_pcd)

    # 执行ICP配准
    transform = icp_registration(source_pcd, target_pcd)

    # 应用配准结果，整合点云并可视化
    source_pcd.transform(transform)
    o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="After ICP Registration")

    # 初始化TSDF体素网格
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # TSDF融合
    integrate_tsdf(source_pcd, transform, volume)

    # 提取并保存融合后的网格模型
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("output_mesh.ply", mesh)

    # 可视化网格模型
    o3d.visualization.draw_geometries([mesh], window_name="Fused Mesh")