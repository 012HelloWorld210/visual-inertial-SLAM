import cv2
import numpy as np

def ransac_match(keypoints1, keypoints2, matches, ransacReprojThreshold=12.0):
    # 获取关键点的坐标
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 使用RANSAC估计几何变换矩阵
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold)
    matchesMask = mask.ravel().tolist()
    # 将RANSAC优化后的点进行保留
    good_matches = [m for m, valid in zip(matches, matchesMask) if valid]
    keypoints1_optimized = np.float32([keypoints1[m.queryIdx].pt for m in good_matches if m.queryIdx < len(keypoints1)])
    keypoints2_optimized = np.float32([keypoints2[m.trainIdx].pt for m in good_matches if m.trainIdx < len(keypoints2)])
    return matchesMask, good_matches, keypoints1_optimized, keypoints2_optimized

# 示例调用
# img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
# orb = cv2.ORB_create()
# keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
# keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descriptors1, descriptors2)
# matches = sorted(matches, key=lambda x: x.distance)
# ransac_matching_and_drawing(img1, img2, keypoints1, keypoints2, matches)
