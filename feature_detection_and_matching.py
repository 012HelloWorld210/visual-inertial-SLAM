import cv2

def match_images(image_path1, image_path2, num_matches=10):
    """
    使用ORB特征检测和BFMatcher进行图像匹配。

    参数：
    - image_path1: str，第一个图像的文件路径。
    - image_path2: str，第二个图像的文件路径。
    - num_matches: int，显示的前几个匹配结果，默认值为10。

    返回：
    - img_matches: 匹配结果的图像。
    """
    # 读取图像
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # 初始化ORB检测器
    orb = cv2.ORB_create()

    # 检测关键点和描述符
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # 创建BFMatcher对象
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 进行匹配
    matches = bf.match(descriptors1, descriptors2)

    # 按照距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:num_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

# 示例调用
image_path1 = 'image1.jpg'
image_path2 = 'image2.jpg'
matched_image = match_images(image_path1, image_path2)

# 显示结果
cv2.namedWindow('RANSAC Matches', cv2.WINDOW_NORMAL)

# 调整窗口大小
cv2.resizeWindow('RANSAC Matches', 1000, 1000)

# 显示图像
cv2.imshow('RANSAC Matches', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
