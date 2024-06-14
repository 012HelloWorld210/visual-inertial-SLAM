import cv2
import numpy as np


img1 = cv2.imread("image1.jpg", 0)
img2 = cv2.imread("image2.jpg", 0)

orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 12.0)
matchesMask = mask.ravel().tolist()
# 此处获取RANSAC优化后的点作为姿态估计的输入
good_matches = [m for m, valid in zip(matches, matchesMask) if valid]
keypoints1_optimized = np.float32([keypoints1[m.queryIdx].pt for m in good_matches if m.queryIdx < len(keypoints1)])
keypoints2_optimized = np.float32([keypoints2[m.trainIdx].pt for m in good_matches if m.trainIdx < len(keypoints2)])
K = np.array([[800, 0, 320],
              [0, 800, 240],
              [0, 0, 1]], dtype=np.float32)
R, t = estimate_pose_from_matches(good_matches, keypoints1_optimized, keypoints2_optimized)

# 投影矩阵
P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(K, np.hstack((R, t)))

# 三角化
points1_h = cv2.convertPointsToHomogeneous(points1)[:, 0, :]
points2_h = cv2.convertPointsToHomogeneous(points2)[:, 0, :]

points_4d_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)

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
