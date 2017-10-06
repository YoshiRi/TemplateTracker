import cv2
import sys

import numpy as np
import matplotlib.pyplot as plt
import time # time.time() to get time



if __name__ == '__main__' :
 
    video = cv2.VideoCapture(0)
 
    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()
         

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        
        cv2.imshow("Current image",frame)
        
        # Exit if q pressed
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') :    
            break
        if k == ord('s') :
            temp = cv2.selectROI(frame)
            cv2.imwrite('object.png',temp)    
            break

