import cv2 as cv


def rescale(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    return resized

capture = cv.VideoCapture('../Resources/Videos/kitten.mp4')

while True:
    ret, frame = capture.read()

    frame_resized = rescale(frame,0.5)
    cv.imshow('resized', frame_resized)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break


capture.release()
cv.destroyAllWindows()