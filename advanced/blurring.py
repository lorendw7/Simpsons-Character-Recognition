import cv2 as cv


# 1. 读取图片
img = cv.imread('../Resources/Photos/cats.jpg')

# 2. 显示原始图片
cv.imshow('cats', img)

# 3. 平均模糊 (Averaging Blur)
average = cv.blur(img, ksize=(3, 3))
cv.imshow('average', average)

# 4. 高斯模糊 (Gaussian Blur)
gaussian = cv.GaussianBlur(img, ksize=(3, 3), sigmaX=0)
cv.imshow('gaussian', gaussian)

# 5. 中值模糊 (Median Blur)
median = cv.medianBlur(img, ksize=3)
cv.imshow('median', median)

# 6. 双边滤波 (Bilateral Filtering)
bilateral = cv.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
cv.imshow('bilateral', bilateral)

# 等待按键后关闭窗口
cv.waitKey(0)