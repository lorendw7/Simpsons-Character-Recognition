import cv2 as cv


# 从指定路径读取一张彩色图片，OpenCV默认以BGR格式读取
img = cv.imread("../Resources/Photos/group 1.jpg")
# 创建一个名为"Photo"的窗口，显示原始彩色图像
cv.imshow("Photo", img)

# 将彩色图像转换为灰度图像，Haar特征检测需要在灰度图上进行
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 创建一个名为"Gray Photo"的窗口，显示灰度图像
cv.imshow("Gray Photo", gray)

# 加载预训练的Haar级联分类器，用于人脸检测
# 'haar_face.xml'是OpenCV提供的人脸检测模型文件
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# 使用detectMultiScale方法在灰度图中检测人脸
# scaleFactor=1.1: 每次图像尺寸减小的比例为10%，用于检测不同大小的人脸
# minNeighbors=6: 每个候选矩形需要保留的邻居数，值越大，检测越严格，误报越少
face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

# 打印检测到的人脸数量，即返回的矩形框列表的长度
print(f'number of faces: {len(face_rect)}')

# 遍历所有检测到的人脸矩形框
for (x,y,w,h) in face_rect:
    # 在原始彩色图像上绘制绿色矩形框，框出人脸
    # (x,y)是矩形左上角坐标，(x+w,y+h)是右下角坐标
    # 颜色为(0,255,0)（绿色），线条宽度为2
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

# 创建一个名为"Face"的窗口，显示标记了人脸框的图像
cv.imshow('Face', img)

cv.waitKey(0)
cv.destroyAllWindows()