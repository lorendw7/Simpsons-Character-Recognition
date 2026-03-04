import cv2 as cv
import numpy as np

# 从指定路径读取彩色图像，OpenCV默认以BGR格式读取
img = cv.imread('../Resources/Photos/park.jpg')
# 创建一个名为'Park'的窗口，显示原始彩色图像
cv.imshow('Park', img)

# 将彩色图像转换为灰度图像，便于后续边缘检测等操作
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 创建一个名为'Gray'的窗口，显示灰度图像
cv.imshow('Gray', gray)

# -------------------------- Laplacian 边缘检测 --------------------------
# 使用Laplacian算子检测图像中的边缘，输出为64位浮点型图像
lap = cv.Laplacian(gray, cv.CV_64F)
# 将浮点型图像转换为8位无符号整型（0-255），并取绝对值以处理负值
lap = np.uint8(np.absolute(lap))
# 创建一个名为'Laplacian'的窗口，显示Laplacian边缘检测结果
cv.imshow('Laplacian', lap)

# -------------------------- Sobel 边缘检测 --------------------------
# 计算x方向（水平方向）的Sobel梯度，dx=1, dy=0
sobelx = cv.Sobel(gray, cv.CV_64F, 1,  0)
# 计算y方向（垂直方向）的Sobel梯度，dx=0, dy=1
sobely = cv.Sobel(gray, cv.CV_64F,0, 1)
# 将x方向和y方向的梯度图像进行按位或操作，得到综合的Sobel边缘检测结果
combined_sobel = cv.bitwise_or(sobelx, sobely)

# 显示x方向的Sobel梯度图像
cv.imshow('Sobel X', sobelx)
# 显示y方向的Sobel梯度图像
cv.imshow('Sobel Y', sobely)
# 显示综合的Sobel边缘检测结果
cv.imshow('Combined Sobel', combined_sobel)

# -------------------------- Canny 边缘检测 --------------------------
# 使用Canny算子进行边缘检测，设置低阈值150和高阈值175
canny = cv.Canny(gray, 150, 175)
# 创建一个名为'Canny'的窗口，显示Canny边缘检测结果
cv.imshow('Canny', canny)

# 等待用户按下任意键后关
cv.waitKey(0)
cv.destroyAllWindows()