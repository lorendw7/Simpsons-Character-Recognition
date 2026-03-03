import cv2 as cv

img = cv.imread('../Resources/Photos/cat.jpg')
cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

canny = cv.Canny(img, 125, 175)
cv.imshow('Canny', canny)

dilated = cv.dilate(canny, (7, 7), iterations=3)
cv.imshow('Dilated', dilated)

eroded = cv.erode(dilated, (3, 3))
cv.imshow('Eroded', eroded)

resized = cv.resize(img, (600, 600), interpolation=cv.INTER_LINEAR)
cv.imshow('Resized', resized)

cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)