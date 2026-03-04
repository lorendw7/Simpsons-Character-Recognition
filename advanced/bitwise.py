import cv2 as cv          # 导入OpenCV库，简写为cv
import numpy as np        # 导入数值计算库，用于创建图像矩阵

# 1. 创建空白画布：400x400像素的单通道灰度图，像素值初始化为0（纯黑色）
blank = np.zeros(shape=(400, 400), dtype='uint8')

# 2. 绘制实心矩形：基于空白画布的副本绘制（避免修改原画布）
# 参数：画布、左上角坐标(30,30)、右下角坐标(370,370)、颜色255（纯白色）、厚度-1（实心）
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)

# 3. 绘制实心圆形：基于空白画布的副本绘制
# 参数：画布、圆心(200,200)、半径200、颜色255（纯白色）、厚度-1（实心）
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

# 4. 显示原始图形
cv.imshow('rectangle', rectangle)  # 显示矩形窗口
cv.imshow('circle', circle)        # 显示圆形窗口

# 5. 位运算：与（AND）
bitwise_and = cv.bitwise_and(rectangle, circle)  # 仅保留两个图形的**重叠区域**
cv.imshow('bitwise_and', bitwise_and)

# 6. 位运算：或（OR）
bitwise_or = cv.bitwise_or(rectangle, circle)    # 保留两个图形的**所有区域**（重叠+非重叠）
cv.imshow('bitwise_or', bitwise_or)

# 7. 位运算：异或（XOR）
bitwise_xor = cv.bitwise_xor(rectangle, circle)  # 仅保留两个图形的**非重叠区域**
cv.imshow('bitwise_xor', bitwise_xor)

# 8. 非运算 非 （NOT）
bitwise_not = cv.bitwise_not(rectangle)  # 图像反转（白变黑，黑变白）
cv.imshow('bitwise_not', bitwise_not)

# 等待按键输入后，关闭所有窗口（0表示无限等待）
cv.waitKey(0)
cv.destroyAllWindows()  # 规范写法