import numpy as np
import cv2


def estimate_pose_from_matches(matches, keypoints1, keypoints2):
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0, 0, 1]], dtype=np.float32)

    matches_opt = matches[:100]  # 使用前100个匹配点

    # 计算基础矩阵 F
    F, mask = cv2.findFundamentalMat(keypoints1, keypoints2, cv2.FM_RANSAC)

    if F is not None:
        # 计算本质矩阵 E
        E = K.T @ F @ K

        if E is not None:
            # 从本质矩阵恢复相对姿态（旋转矩阵 R 和平移向量 t）
            _, R, t, _ = cv2.recoverPose(E, keypoints1, keypoints2, K)
            return R, t

    return None, None, None, None

# 示例调用
# img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
# orb = cv2.ORB_create()
# keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
# keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descriptors1, descriptors2)
# matches = sorted(matches, key=lambda x: x.distance)
# R, t, pts1, pts2 = estimate_pose_from_matches(matches, keypoints1, keypoints2)
