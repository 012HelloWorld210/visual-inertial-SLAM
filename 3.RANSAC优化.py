# 程序主要在特征检测与匹配的基础上，使用RANSAC对误匹配进行优化
import cv2
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
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
M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 12.0)
    # - M：返回的单映射矩阵
    # - mask：匹配点中属于内点的将被标记为1
matchesMask = mask.ravel().tolist()
    # 改变了原本mask的排列顺序，将 `mask` 数组展平成一维数组，然后转换成一个 Python 列表。

# 仅绘制内点
draw_params = dict(matchColor = (0, 255, 0),  # 内点为绿色
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2)
    # - 将绘制所需要的参数写成一个词典
    # 1. matchColor: 指定匹配点的颜色。在你的代码中 `(0, 255, 0)` 表示绿色。如果匹配点是内点，它们会被绘制为绿色。
    # 2. singlePointColor: 指定单个点（未形成匹配）的颜色。`None` 表示使用默认颜色。
    # 3. matchesMask: 这是一个掩码数组，用于指定哪些点需要绘制。只有掩码为 `1` 的匹配对才会被绘制。如果 `matchesMask` 中的值为 `0`，则对应的点将不会被绘制。
    # 4. flags: 控制绘制单点和匹配对的方式。在你的代码中，`flags = 2` 表示只绘制匹配的线段，而忽略单点。
img_ransac_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, **draw_params)
    # - `img1`：输入的第一幅图像。
    # - `keypoints1`：第一幅图像中的关键点集合。
    # - `img2`：输入的第二幅图像。
    # - `keypoints2`：第二幅图像中的关键点集合。
    # - `matches`：两幅图像之间匹配对集合，通常是通过匹配算法（例如 BFMatcher 或 FLANN）获得的。
    # - `None`：无意义参数，此处通常给None即可。
    # - `**draw_params`：可选的绘制参数集，允许指定一些绘制相关的选项，比如颜色、标注等
# 创建一个命名窗口
cv2.namedWindow('RANSAC Matches', cv2.WINDOW_NORMAL)

# 调整窗口大小
cv2.resizeWindow('RANSAC Matches', 800, 800)

# 显示图像
cv2.imshow('RANSAC Matches', img_ransac_matches)

cv2.waitKey(0)
cv2.destroyAllWindows()
