# 导入必要的库
import cv2 as cv
import numpy as np
import os

# 定义需要识别的人员名单（标签对应的人名）
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
# 训练数据集的根目录（r表示原始字符串，避免转义字符问题）
DIR = r"../Resources/Faces/train"

# 加载预训练的Haar级联分类器，用于检测人脸区域
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# 初始化两个空列表：
# features - 存储提取的人脸特征（裁剪后的人脸灰度图）
# labels - 存储对应人脸的标签（与people列表索引对应）
features = []
labels = []

# 定义训练数据创建函数：遍历数据集，提取人脸特征和对应标签
def create_train():
    # 遍历每个人名
    for person in people:
        # 拼接当前人物的图片文件夹路径
        path = os.path.join(DIR, person)
        # 获取当前人物对应的标签（即其在people列表中的索引）
        label = people.index(person)

        # 遍历当前人物文件夹下的所有图片文件
        for img in os.listdir(path):
            # 拼接每张图片的完整路径
            img_path = os.path.join(path, img)

            # 读取图片（OpenCV默认以BGR格式读取）
            img_array = cv.imread(img_path)
            # 检查图片是否读取成功（避免损坏/不存在的图片导致报错）
            if img_array is None:
                continue  # 跳过读取失败的图片

            # 将彩色图片转换为灰度图（人脸检测和识别更适合在灰度图上进行）
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # 使用Haar分类器检测灰度图中的人脸，返回人脸矩形框坐标(x,y,w,h)
            # scaleFactor=1.1：图像金字塔缩放比例，每次缩小10%，适配不同大小的人脸
            # minNeighbors=4：候选框需要满足的邻居数，平衡检测精度和召回率
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            # 遍历检测到的所有人脸矩形框
            for (x, y, w, h) in faces_rect:
                # 裁剪出人脸区域（ROI：感兴趣区域），仅保留人脸部分
                faces_roi = gray[y:y + h, x:x + w]
                # 将裁剪后的人脸特征添加到features列表
                features.append(faces_roi)
                # 将对应的标签添加到labels列表
                labels.append(label)

# 调用函数，生成训练数据
create_train()

# 打印训练完成提示
print('Training done ---------------')

# 将特征和标签列表转换为NumPy数组（OpenCV的识别器要求输入为数组格式）
# dtype='object'：兼容不同尺寸的人脸ROI（不同图片裁剪出的人脸大小可能不同）
features = np.array(features, dtype='object')
labels = np.array(labels)

# 创建LBPH（局部二值模式直方图）人脸识别器实例
# LBPH是OpenCV中轻量级的人脸识别算法，对光照、姿态变化有一定鲁棒性
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# 使用提取的人脸特征和对应标签训练识别器
face_recognizer.train(features, labels)

# 保存训练好的模型到本地文件（后续可直接加载使用，无需重复训练）
face_recognizer.save('face_trained.yml')
# 保存特征和标签数组（可选，用于后续分析/验证）
np.save('features.npy', features)
np.save('labels.npy', labels)