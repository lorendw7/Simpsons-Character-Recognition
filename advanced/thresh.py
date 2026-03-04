import cv2 as cv

# 读取图片：从相对路径加载彩色图像，OpenCV默认以BGR格式读取
img = cv.imread('../Resources/Photos/cats.jpg')
# 显示原始彩色图像，窗口标题为'Cats'
cv.imshow('Cats', img)

# 将彩色图像转换为灰度图，减少计算量，便于后续阈值处理
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 显示灰度图像，窗口标题为'Gray'
cv.imshow('Gray', gray)

# --- Simple Thresholding 简单阈值处理 ---
# 对灰度图进行简单二值化：像素值>150时设为255（白色），否则设为0（黑色）
# 返回值：阈值(150)和处理后的二值图像
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
# 显示简单二值化后的图像，窗口标题为'Simple Thresholded'
cv.imshow('Simple Thresholded', thresh)

# 对灰度图进行反二值化：像素值>150时设为0（黑色），否则设为255（白色）
threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
# 显示反二值化后的图像，窗口标题为'Simple Thresholded Inverse'
cv.imshow('Simple Thresholded Inverse', thresh_inv)

# --- Adaptive Thresholding 自适应阈值处理 ---
# 自适应阈值处理：根据像素邻域动态计算阈值，适合光照不均匀的图像
# 方法：高斯加权平均；类型：反二值化；块大小11x11；阈值偏移量C=9
adaptive_thresh = cv.adaptiveThreshold(gray, 255,
                                      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY_INV, blockSize=11, C=9)
# 显示自适应阈值处理后的图像，窗口标题为'Adaptive Thresholding'
cv.imshow('Adaptive Thresholding', adaptive_thresh)

# 等待键盘输入（0表示无限等待），按任意键后关闭所有OpenCV窗口
cv.waitKey(0)