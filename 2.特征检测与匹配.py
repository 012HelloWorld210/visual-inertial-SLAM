# 程序主要实现图像特征的检测与匹配
import cv2

### 读取图像
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
    #   - 这两个变量分别保存了读取的 ‘frame1.jpg’ 和 ‘frame2.jpg’ 图像，此处是按灰度图模式读取的图像数据。

### 使用OpenCV提供的特征检测算法，如SIFT、SURF或ORB。
# 初始化ORB检测器
orb = cv2.ORB_create()
    # - 这是一个初始化了的ORB（Oriented FAST and Rotated BRIEF）特征检测器对象，用于检测图像中的特征点和描述子。
    # - 此处也可替换为其他检测方法
# 检测特征点和计算描述子
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    #  - `keypoints1` 和 `keypoints2` 分别是 `img1` 和 `img2` 图像中的特征点。每个特征点包含其位置、尺度、角度等信息。
    #  - `descriptors1` 和 `descriptors2` 是特征点的描述子，它们是用于描述图像中特征点局部图像块的向量。ORB描述子是二进制字符串。

### 使用特征匹配器（例如BFMatcher）来匹配两个图像之间的特征。
# 使用汉明距离的BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # - ‘bf’ 是一个BFMatcher（Brute-Force Matcher）对象，用于执行图像特征匹配，这里指定了使用汉明距离（cv2.NORM_HAMMING）进行描述子匹配，并启用了交叉检查（crossCheck=True）
# 进行匹配
matches = bf.match(descriptors1, descriptors2)
    #  `bf.match(descriptors1, descriptors2)`的输出是一个包含`DMatch`对象的列表。每个`DMatch`对象有以下属性：
    # 1. `queryIdx`：查询描述符（即第一个图像的描述符）的索引。
    # 2. `trainIdx`：训练描述符（即第二个图像的描述符）的索引。
    # 3. `imgIdx`：训练图像的索引（在这种情况下通常是0，因为我们在只有两个图像的情况下使用）。
    # 4. `distance`：描述符之间的距离（越小表示匹配越好）
# 根据距离排序
matches = sorted(matches, key=lambda x: x.distance)
    #将 `matches` 列表中的元素按照其 `distance` 属性值从小到大排序。

###在两幅图像上绘制匹配结果，以便可视化检查
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# 创建一个命名窗口
cv2.namedWindow('RANSAC Matches', cv2.WINDOW_NORMAL)

# 调整窗口大小
cv2.resizeWindow('RANSAC Matches', 800, 800)

# 显示图像
cv2.imshow('RANSAC Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

### 附加内容
#查看keypoint的部分信息
for keypoint in keypoints1:
    x = keypoint.pt[0]  # 获取x坐标
    y = keypoint.pt[1]  # 获取y坐标
    size = keypoint.size  # 获取特征点大小
    angle = keypoint.angle  # 获取特征点角度

    print(f"坐标: ({x}, {y}), 大小: {size}, 角度: {angle}")
# 查看descriptors1的部分信息
print("Descriptors:\
", descriptors1)
    # 此处的描述子应该为一系列 0 1序列