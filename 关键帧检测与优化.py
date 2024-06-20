import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R

# 初始化ORB特征提取器
orb = cv2.ORB_create()

# 定义Bag of Words模型
class BoW:
    def __init__(self, n_clusters=100):
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.vocab = None

    def fit(self, descriptors):
        all_descriptors = np.vstack(descriptors)
        self.kmeans.fit(all_descriptors)
        self.vocab = self.kmeans.cluster_centers_

    def transform(self, descriptors):
        words = self.kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(len(self.vocab) + 1))
        return hist

# 提取特征和描述子
def extract_features(image):
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# RANSAC进行几何验证
def ransac_homography(matches, kp1, kp2):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    return M, matchesMask

# 位姿图优化
class PoseGraphOptimization:
    def __init__(self):
        self.poses = []
        self.edges = []

    def add_vertex(self, id, pose, fixed=False):
        if fixed:
            self.poses.append(np.zeros(6))
        else:
            self.poses.append(pose)

    def add_edge(self, id1, id2, relative_pose, information=np.identity(6)):
        self.edges.append((id1, id2, relative_pose, information))

    def optimize(self, max_iterations=10):
        for _ in range(max_iterations):
            for (id1, id2, relative_pose, information) in self.edges:
                pose1 = self.poses[id1]
                pose2 = self.poses[id2]
                updated_pose2 = pose1 + relative_pose  # Simplified update step
                self.poses[id2] = updated_pose2

# 主程序
def main():
    # 读取图像数据
    images = [cv2.imread(f'image_{i}.png', 0) for i in range(5)]

    # 提取特征和描述子
    descriptors = []
    for img in images:
        _, des = extract_features(img)
        descriptors.append(des)

    # 构建BoW模型
    bow = BoW(n_clusters=100)
    bow.fit(descriptors)
    histograms = [bow.transform(des) for des in descriptors]

    # 初始化匹配器
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 初始化位姿图优化器
    pgo = PoseGraphOptimization()

    # 添加第一个节点
    pgo.add_vertex(0, np.zeros(6), fixed=True)

    # 逐帧处理
    for i in range(1, len(images)):
        # 匹配描述子
        matches = bf.match(descriptors[i - 1], descriptors[i])
        matches = sorted(matches, key=lambda x: x.distance)

        # RANSAC几何验证
        kp1, _ = extract_features(images[i - 1])
        kp2, _ = extract_features(images[i])
        M, matchesMask = ransac_homography(matches, kp1, kp2)

        # 假设相对变换矩阵为单位矩阵
        relative_pose = np.hstack((np.zeros(3), R.from_matrix(np.eye(3)).as_quat()))

        # 添加节点和边
        pgo.add_vertex(i, np.zeros(6))
        pgo.add_edge(i - 1, i, relative_pose)

    # 进行优化
    pgo.optimize()

if __name__ == '__main__':
    main()
