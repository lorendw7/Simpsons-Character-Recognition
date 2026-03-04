import cv2 as cv
import numpy as np

# 1. 读取并显示原始猫咪图片
img = cv.imread('../Resources/Photos/cats 2.jpg')
cv.imshow('Cats', img)

# 2. 创建与原图尺寸相同的空白画布（单通道灰度图，全黑）
# img.shape[:2] 取图像的高度和宽度，确保画布与原图大小一致
blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank Image', blank)

# 3. 在空白画布上画一个白色实心圆
# 圆心：图像中心向右偏移45像素；半径：100像素；颜色：255（白色）；厚度：-1（实心）
circle = cv.circle(blank.copy(),
                   (img.shape[1]//2 + 45, img.shape[0]//2),
                   100,
                   255,
                   -1)

# 4. 在空白画布上画一个白色实心矩形
# 左上角(30,30)，右下角(370,370)，颜色255，实心
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)

# 5. 用“与运算”得到圆和矩形的重叠区域，作为最终的遮罩（weird_shape）
# 只有同时是白色（255）的区域才会保留为白色，其余为黑色
weird_shape = cv.bitwise_and(circle, rectangle)
cv.imshow('Weird Shape', weird_shape)

# 6. 用遮罩（weird_shape）对原图进行“抠图”
# 只有遮罩中白色（255）的区域，原图的对应像素才会被保留；黑色区域则变为黑色
masked = cv.bitwise_and(img, img, mask=weird_shape)
cv.imshow('Weird Shaped Masked Image', masked)

# 等待按键后关闭所有窗口
cv.waitKey(0)