# 程序主要测试设备摄像头是否存在问题
# 前置条件：环境中安装 openCV 模块--pip install opencv-contrib-python

import cv2
# 打开视频设备
cap = cv2.VideoCapture(0)
    #   - `cap`：这是一个cv2.VideoCapture对象，用来从系统默认摄像头（设备编号0）打开视频捕捉。
while True:
    # 读取帧
    ret, frame = cap.read()
        #   - `ret`：这是一个布尔值，表示是否成功读取了一帧。成功读取为True，失败为False。
        #   - `frame`：这是当前捕捉到的一帧图像，类型为ndarray。

    # 显示帧
    cv2.imshow('Frame', frame)
        #  - `'Frame'`：这是窗口的名字。
        #  - `frame`：这是要在窗口中显示的帧图像。

    # 按下q键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        #    - `cv2.waitKey(1)`：这将等待1毫秒以检查是否有键盘输入。
        #    - `cv2.waitKey(1) & 0xFF`：处理了按键值，得到键盘按键的ASCII码。
        #    - `ord('q')`：得到字符'q'的ASCII码，可以与按下的按键进行比较。
        #    - 如果按下了'q'键，则条件为真，结束循环。

# 释放视频设备和关闭窗口
cap.release()
cv2.destroyAllWindows()