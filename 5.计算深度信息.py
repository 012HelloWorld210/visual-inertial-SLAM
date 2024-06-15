import cv2
import numpy as np
from feature_detection_and_matching import match_images
from ransac_optimization import ransac_match
from calculate_camera_pose import estimate_pose_from_matches

img1_path = "image1.jpg"
img2_path = "image2.jpg"
# 进行特征匹配与检测
img1, keypoints1, img2, keypoints2, matches, img_matches = match_images(img1_path,img2_path)
# 使用ransac计算单映射矩阵并保存优化结果
match_mask,good_matches, keypoints1_optimized, keypoints2_optimized= ransac_match(keypoints1, keypoints2, matches)
# 给出相机内参矩阵
K = np.array([[800, 0, 320],
              [0, 800, 240],
              [0, 0, 1]], dtype=np.float32)
# 使用优化后的结果计算相机的位姿
R, t= estimate_pose_from_matches(good_matches, keypoints1_optimized, keypoints2_optimized)

# 投影矩阵
P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(K, np.hstack((R, t)))

# 三角化
points1_h = cv2.convertPointsToHomogeneous(keypoints1_optimized)[:, 0, :]
points2_h = cv2.convertPointsToHomogeneous(keypoints2_optimized)[:, 0, :]

points_4d_hom = cv2.triangulatePoints(P1, P2, keypoints1_optimized.T, keypoints2_optimized.T)

# 转换为非齐次坐标
points_3d = points_4d_hom / points_4d_hom[3]
points_3d = points_3d[:3].T

# 打印或保存3D点云
print("3D points:\n", points_3d)

# 此处需要安装matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维点云
ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c='b', marker='o')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置坐标轴刻度
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# 显示图形
plt.show()
