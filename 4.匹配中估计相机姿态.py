# 程序主要实现 计算基础矩阵或本质矩阵、从本质矩阵恢复相机姿态
import numpy as np
import cv2

# 计算基础矩阵 F（Fundamental Matrix）
def compute_fundamental_matrix(matches, keypoints1, keypoints2):
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return F, mask

# 计算本质矩阵 E（Essential Matrix）
def compute_essential_matrix(F, K):
    E = K.T @ F @ K
    return E

# 从本质矩阵恢复相对姿态（旋转矩阵 R 和平移向量 t）
def recover_pose(E, pts1, pts2, K):
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

# 主函数
def estimate_pose_from_matches(matches, keypoints1, keypoints2):
    # 相机内参矩阵 K（由相机标定获得）
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0, 0, 1]], dtype=np.float32)

    matches_opt = matches[:100]  # 使用前100个匹配点
    F, mask = compute_fundamental_matrix(matches_opt, keypoints1, keypoints2)

    if F is not None:
        E = compute_essential_matrix(F, K)
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches_opt])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches_opt])

        if E is not None:
            R, t = recover_pose(E, pts1, pts2, K)
            return R, t

    return None, None