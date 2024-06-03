# 程序主要在特征检测与匹配的基础上，使用RANSAC对误匹配进行优化
import cv2
img1 = cv2.imread('frame1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('frame2.png', cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

### 从此处开始进行RANSAC的实现
# 首先要安装 numpy库 pip install numpy
import numpy as np

# 获取关键点的坐标
pts1 = np.float32([ keypoints1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
pts2 = np.float32([ keypoints2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    # 此处首先从 matches 获取两张图像匹配的 keypoints，接着读取匹配的坐标，然后转化为浮点型，最后调节为若干个1×2的矩阵。 
# 使用RANSAC估计几何变换矩阵
M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()
print(matchesMask)
# 仅绘制内点
draw_params = dict(matchColor = (0, 255, 0),  # 内点为绿色
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)

img_ransac_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, **draw_params)

cv2.imshow('RANSAC Matches', img_ransac_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
