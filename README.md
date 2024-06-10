# 从零手搓slam

## 说明

​	额，前期主要是vslam。作为一个初入slam的小白，大部分时间只能调用现成的开源框架，如orb-slam、vin-fusion等，因为其是完整的框架，所以调用起来非常方便，但是整体理解起来比较抽象，主要是因为框架要考虑的内容较多，所以不好找vslam理论中每块在框架中的哪一块，尤其是在刚入门研究vslam理论基础时，更是头大。

​	因此，本文仅从vslam的理论研究开始，一步步进行实现，不考虑框架所注意那么多东西。

## 概述

<img src=".\image\主程序.png" alt="主程序" style="zoom: 33%;" />

​	正常处理流程图如上图所示，而本次从零手搓slam也将按照如上流程图进行。

## 初始化

<img src=".\image\初始化子程序.png" style="zoom: 33%;" />

初始化子程序如上图所示，其中，相机预处理为：

1. **内参 (Intrinsic Parameters)**

​	   \- 焦距 (focal length) 

​	   - 光心坐标 (principal point)

​           - 畸变系数 (distortion coefficients)

2. **外参 (Extrinsic Parameters)**：

​	\- 旋转矩阵 (rotation matrix)

​	\- 平移向量 (translation vector)

3. **相机型号信息**：

​	\- 传感器尺寸 (sensor size)

​	\- 镜头像素比例 (pixel aspect ratio)

imu预处理为：

1. **内参 (Intrinsic Parameters)**：

​	\- 陀螺仪偏置 (gyroscope bias)

​	\- 加速度计偏置 (accelerometer bias)

​	\- 陀螺仪噪声 (gyroscope noise)

​	\- 加速度计噪声 (accelerometer noise)

2. **外参 (Extrinsic Parameters)**：

​	\- 安装位置相对于其他传感器的旋转矩阵 (rotation matrix relative to other sensors)

​	\- 安装位置相对于其他传感器的平移向量 (translation vector relative to other sensors)

3. **其他校准参数**：

​	\- 比如温度补偿系数 (temperature compensation coefficients) 等等。