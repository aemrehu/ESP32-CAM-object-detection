import cv2
import urllib.request
import numpy as np

url = 'http://192.168.10.42:81'
cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
