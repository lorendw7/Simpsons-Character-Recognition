import cv2 as cv

# img = cv.imread("./Resources/Photos/cat_large.jpg")
# cv.imshow("cat", img)
# cv.waitKey(0)

capture = cv.VideoCapture('../Resources/Videos/kitten.mp4')

while True:
    ret, frame = capture.read()
    cv.imshow('frame', frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()