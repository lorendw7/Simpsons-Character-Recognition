# 导入OpenCV库，并使用别名cv方便后续调用
import cv2 as cv

# 定义需要识别的人员名单（标签对应的人名）
# 列表索引与训练时的标签一一对应（如0对应Ben Afflek，1对应Elton John等）
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# 加载预训练的Haar级联分类器，用于检测图片中的人脸区域
# 该文件是OpenCV提供的人脸检测模型，需确保文件路径正确
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# 创建LBPH（局部二值模式直方图）人脸识别器实例
face_recognizer = cv.face.LBPHFaceRecognizer_create()
# 加载之前训练好的人脸识别模型文件（face_trained.yml）
# 该文件包含了训练好的人脸特征和标签映射关系，无需重复训练
face_recognizer.read('face_trained.yml')

# 读取待识别的测试图片（使用r前缀避免转义字符问题）
# 路径指向验证集里Elton John的一张测试图片
img = cv.imread(r'../Resources\Faces\val\elton_john/1.jpg')

# 将彩色测试图片转换为灰度图（人脸检测和识别均基于灰度图）
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 显示灰度化后的图片，窗口名称为'Person'
cv.imshow('Person', gray)

# 第一步：检测图片中的人脸区域
# 使用Haar分类器检测灰度图中的人脸，返回人脸矩形框坐标(x,y,w,h)
# 参数说明：1.1=缩放比例（每次缩小10%），4=最小邻居数（控制检测精度）
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

# 遍历所有检测到的人脸矩形框（一张图可能有多人脸）
for (x,y,w,h) in faces_rect:
    # 裁剪出人脸区域（ROI：感兴趣区域），仅保留人脸部分用于识别
    faces_roi = gray[y:y+h, x:x+w]

    # 第二步：使用训练好的模型识别人脸
    # predict方法返回两个值：label（预测标签）、confidence（置信度，值越低匹配度越高）
    label, confidence = face_recognizer.predict(faces_roi)
    # 打印识别结果：标签对应的人名 + 置信度
    print(f'Label = {people[label]} with a confidence of {confidence}')

    # 在原始彩色图片上绘制识别结果文本
    # 参数：图片、文本位置(20,20)、字体、字体大小1.0、颜色绿色(0,255,0)、线条粗细2
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    # 在原始彩色图片上绘制人脸矩形框，框出识别的人脸
    # 参数：图片、矩形左上角(x,y)、右下角(x+w,y+h)、颜色绿色、线条粗细2
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

# 显示绘制了识别结果的图片，窗口名称为'Detected Face'
cv.imshow('Detected Face', img)

# 等待用户按下任意键后关闭所有窗口（0表示无限等待）
cv.waitKey(0)