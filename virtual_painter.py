import cv2
import numpy as np
canvas = np.zeros((480, 640, 3), dtype=np.uint)
cv2.imshow("Canvas", canvas)
cv2.waitKey(1)
cv2.destroyAllWindows()

