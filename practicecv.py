import cv2

cap = cv2.VideoCapture(0)

success, frame = cap.read()

print(success)
# True

print(frame.shape)
# (480, 640, 3)  Example only