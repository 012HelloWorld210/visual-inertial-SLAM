import numpy as np
import cv2

# 计算基础矩阵 F
def compute_fundamental_matrix(keypoints1, keypoints2):
    F, mask = cv2.findFundamentalMat(keypoints1, keypoints2, cv2.FM_RANSAC)
    return F, mask
    # -- 解算基础矩阵需要来自两个坐标系的坐标
# 计算本质矩阵 E
def compute_essential_matrix(F, K):
    E = K.T @ F @ K
    return E
    # --本质矩阵相当于将将相机内参矩阵加入基础矩阵，是其的一种特例情况
# 从本质矩阵恢复相对姿态（旋转矩阵 R 和平移向量 t）
def recover_pose(E, pts1, pts2, K):
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t
    # -- 此处的姿态解算原理为 同一位置在不同的坐标系下都可以通过平移和旋转进行转换
# 主函数，用于从匹配中估计姿态
def estimate_pose_from_matches(matches, keypoints1, keypoints2):
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0, 0, 1]], dtype=np.float32)

    matches_opt = matches[:100]  # 使用前100个匹配点
    F, mask = compute_fundamental_matrix(keypoints1, keypoints2)

    if F is not None:
        E = compute_essential_matrix(F, K)
        pts1 = keypoints1
        pts2 = keypoints2

        if E is not None and len(pts1) > 0 and len(pts2) > 0:
            R, t = recover_pose(E, pts1, pts2, K)
            return R, t, pts1, pts2

    return None, None

# 主函数调用示例
if __name__ == "__main__":
    img1 = cv2.imread("image1.jpg", 0)
    img2 = cv2.imread("image2.jpg", 0)

    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 12.0)
    matchesMask = mask.ravel().tolist()
    # 此处获取RANSAC优化后的点作为姿态估计的输入
    good_matches = [m for m, valid in zip(matches, matchesMask) if valid]
    keypoints1_optimized = np.float32([keypoints1[m.queryIdx].pt for m in good_matches if m.queryIdx < len(keypoints1)])
    keypoints2_optimized = np.float32([keypoints2[m.trainIdx].pt for m in good_matches if m.trainIdx < len(keypoints2)])

    R, t = estimate_pose_from_matches(good_matches, keypoints1_optimized, keypoints2_optimized)

    if R is not None and t is not None:
        print("Estimated Rotation:", R)
        print("Estimated Translation:", t)
    else:
        print("Failed to estimate pose.")
