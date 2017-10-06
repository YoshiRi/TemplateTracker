import cv2
import sys

import numpy as np
import matplotlib.pyplot as plt
import time # time.time() to get time

class Checktime:
    def __init__(self):
        self.checktime = [0]
        self.difftime = []
        self.start = time.time()
    def check(self,name=""):
        self.checktime.append(time.time()-self.start)
        self.difftime.append(self.checktime[-1]-self.checktime[-2])
        print(name)
        print("Now time is "+str(self.checktime[-1])+" [s]")
        print("Computed in "+str(self.difftime[-1])+" [s]\n")
        
    def show(self):
        leng = len(self.checktime)
        #plt.plot(np.arange(0,leng,1),np.array(self.checktime),label="Accumulation time")
        fig = plt.figure()
        plt.plot(np.arange(1,leng,1),np.array(self.difftime),label="Each Process time")
        plt.title("Calculation Time")
        plt.xlabel("Sequence")
        plt.ylabel("Consumed Time [s]")
        plt.ylim(0,1)
        plt.legend()
        plt.grid()
        plt.show()


class TempTracker:

    def __init__(self,temp,descriptor = 'ORB'):
        
        # switch detector and matcher
        self.detector = self.get_des(descriptor)
        self.bf =  self.get_matcher(descriptor)# self matcher
        
        if self.detector == 0:
            print("Unknown Descriptor! \n")
            sys.exit()
        
        if len(temp.shape) > 2: #if color then convert BGR to GRAY
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        
        self.template = temp
        #self.imsize = np.shape(self.template)
        self.kp1, self.des1 = self.detector.detectAndCompute(self.template,None)        

        self.flag = 0 # homography estimated flag
        
    def get_des(self,name):
        return {
            'ORB': cv2.ORB_create(nfeatures=500,scoreType=cv2.ORB_HARRIS_SCORE),
            'AKAZE': cv2.AKAZE_create(),
            'KAZE' : cv2.KAZE_create(),
            'SIFT' : cv2.xfeatures2d.SIFT_create(),
            'SURF' : cv2.xfeatures2d.SURF_create()
        }.get(name, 0)  
    
    def get_matcher(self,name): # Binary feature or not 
        return {
            'ORB'  : cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
            'AKAZE': cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
            'KAZE' : cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
            'SIFT' : cv2.BFMatcher(),
            'SURF' : cv2.BFMatcher()
        }.get(name, 0)  
    
    def track(self,img):
        if len(img.shape) > 2: #if color then convert BGR to GRAY
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
             
        kp2,des2 = self.detector.detectAndCompute(img,None)
        print(len(kp2))
        if len(kp2) < 5:
            return
            
        matches = self.bf.knnMatch(self.des1,des2,k=2)
        good = []
        pts1 = []
        pts2 = []
   
        count = 0
        for m,n in matches:      
            if m.distance < 0.5*n.distance:
                good.append([m])
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(self.kp1[m.queryIdx].pt)
                count += 1

        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)
        
        self.flag = 0
        self.show = img

        if count > 4:
            self.H, self.mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,3.0)
            if self.check_mask():
                self.get_rect()
                self.flag = 1
                 
        cv2.imshow("detected",self.show)
        
    def get_rect(self):
            h,w = self.template.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            self.rect = cv2.perspectiveTransform(pts,self.H)
            # draw lines
            self.show = cv2.polylines(self.show,[np.int32(self.rect)],True,255,3, cv2.LINE_AA)
    
    def check_mask(self):
        self.inliner = np.count_nonzero(self.mask)
        print("inliner : "+str(self.inliner))
        #self.total = self.mask.size
        if self.inliner > 4:
            return 1
        else:
            return 0

if __name__ == '__main__' :
 
    DES = sys.argv[1:] # argument is list
    if DES == []:
        DES = ['ORB']
    DES = "".join(DES)    
    print("Using "+DES+" Detector! \n")
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
     
    # read template
    temp = cv2.imread("object.png")
    cv2.imshow("template",temp)
    
    tracker = TempTracker(temp,DES)
    T = Checktime()

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        
        # Tracking Object
        tracker.track(frame)
        T.check()
        
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') :
            T.show()    
            break
        if k == ord('s') :
            cv2.imwrite('result.png',tracker.show)    
            break
