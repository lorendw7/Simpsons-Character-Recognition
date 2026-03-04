import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取原始彩色图像
# 从指定路径读取图片，OpenCV默认以BGR格式存储
img = cv.imread('../Resources/Photos/cats.jpg')
# 显示原始图像，窗口名为'Cats'
cv.imshow('Cats', img)

# 2. 创建空白画布，用于绘制掩码
# 创建与原图尺寸相同（高、宽）的全黑图像，数据类型为uint8（0-255）
blank = np.zeros(img.shape[:2], dtype='uint8')

# 3. 将彩色图像转换为灰度图
# 直方图分析通常在单通道灰度图上进行，这里将BGR转为GRAY
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 显示灰度图，窗口名为'Gray'
cv.imshow('Gray', gray)

# 4. 在空白画布上绘制圆形掩码
# 在图像正中心绘制一个半径为100的实心白色圆，作为有效分析区域
# 圆心坐标：(图像宽度//2, 图像高度//2)
# 颜色：255（白色，代表掩码有效区域），厚度：-1（实心填充）
circle = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)

# 5. 应用掩码，提取局部图像
# 按位与操作：仅保留灰度图中与掩码白色区域重叠的像素，其余区域置为黑色
masked = cv.bitwise_and(gray, gray, mask=circle)
# 显示掩码处理后的图像，窗口名为'Mask'
cv.imshow('Mask', masked)

# 6. 计算带掩码的灰度直方图
# 只统计掩码有效区域（中心圆形）内的像素灰度分布
# images: 输入图像列表
# channels: 通道索引，灰度图只有一个通道，故为[0]
# mask: 掩码，指定有效分析区域
# histSize: 直方图分箱数，灰度值0-255，故为256
# ranges: 像素值范围
# gray_hist = cv.calcHist(images=[gray], channels=[0], mask=masked, histSize=[256], ranges=[0, 256])
#
# # 7. 使用Matplotlib可视化直方图
# plt.figure()
# # 设置图表标题
# plt.title('Grayscale Histogram')
# # X轴标签：代表灰度值分箱（Bins）
# plt.xlabel('Bins')
# # Y轴标签：代表对应灰度值的像素数量
# plt.ylabel('# of pixels')
# # 绘制直方图曲线
# plt.plot(gray_hist)
# # 限制X轴范围为0-256，与灰度值范围一致
# plt.xlim([0, 256])
# # 显示图表
# plt.show()


colors = ('b', 'g', 'r')
plt.figure()
plt.title('Colors Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()


# 8. 等待用户按键，然后关闭所有窗口
cv.waitKey(0)
cv.destroyAllWindows()