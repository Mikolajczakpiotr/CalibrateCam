import numpy as np
import cv2
import glob
import sys
import pickle

cap = cv2.VideoCapture(0)

with open('workfile.pckl','rb+') as f:
    mtx,dist = pickle.load(f)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    cv2.imshow('Cam',img)

    # Our operations on the frame come here
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Display the resulting frame
    cv2.imshow('img',dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()