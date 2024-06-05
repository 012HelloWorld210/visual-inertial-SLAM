# 1. 导入必要的库
import cv2
import numpy as np
# 2. 定义基本参数
    # 假设我们已经有两个相机的内参和外参
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0, 1]])  # 内参矩阵
dist_coeffs = np.zeros((4, 1))  # 假设无畸变
R1, t1 = np.eye(3), np.zeros((3, 1))  # 第一个相机的旋转矩阵和平移向量（参考系）
R2, t2 = external_params  # 第二个相机的外参