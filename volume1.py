import cv2
import numpy as np
import math
from Quartz.CoreGraphics import CGEventCreate, CGEventSourceCreate, kCGEventSourceStateHIDSystemState
from Quartz import CGDisplayBounds, CGMainDisplayID
import os

def count_fingers(contour, drawing):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0
        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.dist(start, end)
            b = math.dist(start, far)
            c = math.dist(end, far)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

            if angle <= math.pi / 2:
                finger_count += 1
                cv2.circle(drawing, far, 8, [211, 84, 0], -1)
        return finger_count + 1
    return 0

def set_volume(change):
    os.system(f"osascript -e 'set volume output volume (output volume of (get volume settings) {change} 10)'")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    kernel = np.ones((3,3), np.uint8)
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100,100), (400,400), (0,255,0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5,5), 100)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        drawing = np.zeros(roi.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0,255,0), 2)

        fingers = count_fingers(contour, drawing)
        cv2.putText(frame, f'Fingers: {fingers}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        if fingers == 1:
            set_volume("-")
            cv2.putText(frame, 'Volume Down', (50,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        elif fingers >= 5:
            set_volume("+")
            cv2.putText(frame, 'Volume Up', (50,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow('Drawing', drawing)
    except:
        pass

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

