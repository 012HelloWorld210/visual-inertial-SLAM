import cv2
import numpy as np

# 读取图像
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化ORB特征检测器
orb = cv2.ORB_create()

# 检测特征点和计算描述子
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# 初始化FLANN匹配器
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 使用FLANN匹配器进行特征匹配
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 使用Ratio Test进行筛选
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 提取匹配点
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用RANSAC计算单应性矩阵并剔除误匹配
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

# 绘制匹配结果
draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, **draw_params)
# 创建一个命名窗口
cv2.namedWindow('RANSAC Matches', cv2.WINDOW_NORMAL)

# 调整窗口大小
cv2.resizeWindow('RANSAC Matches', 800, 800)

# 显示图像
cv2.imshow('RANSAC Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
