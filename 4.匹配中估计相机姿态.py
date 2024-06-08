# 程序主要实现 计算基础矩阵与本质矩阵、从本质矩阵恢复相机姿态
import numpy as np
import cv2

# 计算基础矩阵 F（Fundamental Matrix）
def compute_fundamental_matrix(matches, keypoints1, keypoints2):
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    return F, mask
    # 1. `pts1`：第一幅图像中的点（二维数组，大小为 Nx2，其中 N 是点的数目，每个点包含 (x, y) 坐标）。
    # 2. `pts2`：第二幅图像中的点，与 `pts1` 的点一一对应（二维数组，尺寸与 `pts1` 相同）。
    # 3. `method`：用于估计核心矩阵的算法：
    #    - `cv2.FM_7POINT`：7 点法
    #    - `cv2.FM_8POINT`：8 点法
    #    - `cv2.FM_RANSAC`：基于 RANSAC 的鲁棒估计方法
    #    - `cv2.FM_LMEDS`：最小中值平方法
    # 1. `F`：计算出的基础矩阵（3x3矩阵）。
    # 2. `mask`：一个掩码，通过 RANSAC 方法标记出内点和外点的各标号，与输入点对应。内点的值为 1，外点的值为 0

# 计算本质矩阵 E（Essential Matrix）
def compute_essential_matrix(F, K):
    E = K.T @ F @ K  # 使用 @ 符号表示矩阵乘法，而不是简单的元素乘法
    return E
    #    - `F`: 基础矩阵（3x3的矩阵），表示图像之间的几何关系。
    #    - `K`: 摄影机的内参矩阵（3x3的矩阵），其中包含摄像头的固有属性参数。
    #    - 这里 `K^T` 表示内参矩阵 `K` 的转置。
    #    - 矩阵乘法用于计算本质矩阵，反映从基础矩阵到本质矩阵的变换，这是一步正变换。
    #    - 返回值 `E`：这是计算得出的本质矩阵，表示从像素坐标到相机坐标的关系。
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
# 主函数调用：主程序入口点，即添加前端测试样例或接口
if __name__ == "__main__":
    # 假组件：替换为实际的检测和描述过程
    img1 = cv2.imread("image1.jpg", 0)
    img2 = cv2.imread("image2.jpg", 0)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    ran_pts1 = pts1[matchesMask == 1]
    ran_pts2 = pts2[matchesMask == 1]
    ran_pts1 = ran_pts1.reshape(-1, 2)
    ran_pts2 = ran_pts2.reshape(-1, 2)

    ransac_keypoints1 = []
    ransac_descriptors1 = []
    ransac_keypoints2 = []
    ransac_descriptors2 = []
    # 保留RANSAC优化后的关键点和描述符
    for i, m in enumerate(matches):
        if matchesMask[i]:
            ransac_keypoints1.append(keypoints1[m.queryIdx])
            ransac_descriptors1.append(descriptors1[m.queryIdx])
            ransac_keypoints2.append(keypoints2[m.trainIdx])
            ransac_descriptors2.append(descriptors2[m.trainIdx])
    # 将描述符转换回numpy数组
    ransac_descriptors1 = np.array(ransac_descriptors1)
    ransac_descriptors2 = np.array(ransac_descriptors2)
    # 调用主要函数来估计姿态
    R, t = estimate_pose_from_matches(matches, ransac_keypoints1, ransac_keypoints2)

    if R is not None and t is not None:
        print("Estimated Rotation:\
", R)
        print("Estimated Translation:\
", t)
    else:
        print("Failed to estimate pose.")