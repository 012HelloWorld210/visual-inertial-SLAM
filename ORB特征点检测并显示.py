import cv2
import matplotlib.pyplot as plt

# 读取图像
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化ORB特征检测器
orb = cv2.ORB_create()
# 检测特征点和计算描述子
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# 在图像上绘制特征点
img1_with_keypoints = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0), flags=0)
img2_with_keypoints = cv2.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0), flags=0)

# 显示图像
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.title('Image 1 with Keypoints')
plt.imshow(img1_with_keypoints, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Image 2 with Keypoints')
plt.imshow(img2_with_keypoints, cmap='gray')
plt.axis('off')

plt.show()
