import cv2

def draw_ransac_matches(img1, keypoints1, img2, keypoints2, matches, matchesMask):
    draw_params = dict(matchColor=(0, 255, 0),  # 内点为绿色
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    img_ransac_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, **draw_params)
    return img_ransac_matches

def show_image(window_name, img, width=1000, height=1000):
    # 创建一个命名窗口
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 调整窗口大小
    cv2.resizeWindow(window_name, width, height)

    # 显示图像
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

