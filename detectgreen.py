import cv2
import numpy as np


def nothing(x) -> object:
    pass


def mouse_callback(event, x, y, flags, params):
    if event == 2:
        print(f"coords {x, y}, colors Blue- {img[y, x, 0]} , Green- {img[y, x, 1]}, Red- {img[y, x, 2]} ")


# cap = cv2.VideoCapture(0)
cv2.namedWindow("control")
cv2.createTrackbar("Lower R", "control", 0, 255, nothing)
cv2.createTrackbar("Lower G", "control", 0, 255, nothing)
cv2.createTrackbar("Lower B", "control", 0, 255, nothing)

cv2.createTrackbar("Upper R", "control", 0, 255, nothing)
cv2.createTrackbar("Upper G", "control", 0, 255, nothing)
cv2.createTrackbar("Upper B", "control", 0, 255, nothing)

cv2.setMouseCallback("result", mouse_callback)


img = cv2.imread("img_1.png")

hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
hsv = cv2.GaussianBlur(hsv, (7, 7), 0)
while True:
    # _, img = cap.read()
    lr = cv2.getTrackbarPos("Lower R", "control")
    lg = cv2.getTrackbarPos("Lower G", "control")
    lb = cv2.getTrackbarPos("Lower B", "control")

    ur = cv2.getTrackbarPos("Upper R", "control")
    ug = cv2.getTrackbarPos("Upper G", "control")
    ub = cv2.getTrackbarPos("Upper B", "control")

    lower = np.array([32, 49, 47])
    upper = np.array([101, 255, 255])
    # lower = np.array([lr, lg, lb])
    # upper = np.array([ur, ug, ub])
    imga = img.copy()
    thresh = cv2.inRange(hsv, lower, upper)
    # result = cv2.bitwise_and(img, img, mask=mask)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            imga = cv2.drawContours(imga, [cnt], 0, (0, 255, 255), 4)
            # imga = cv2.rectangle(imga, (x, y), (x + w, y + h), (0, 0, 0), 4)

    cv2.imshow("result", imga)
    cv2.imwrite("test.png", imga)

    if cv2.waitKey(1) == 0x27:
        break

    break

