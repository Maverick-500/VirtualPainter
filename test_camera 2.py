import cv2
import time

#Open the Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open Camera")
    exit()

start_time = time.time()
duration = 10 #seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    #Display the frame
    cv2.imshow("Test Camera", frame)

    #Exit if 'q' is pressed or time is exceeds duration
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    if time.time() - start_time > duration:
        break

#Release Resources
cap.release()
cv2.destroyAllWindows()
print("Test finished: camera was visible for 10 seconds")
