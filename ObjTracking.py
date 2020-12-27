import cv2
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import time # time.time() to get time

# read image in japanese direcotry
def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

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
    """
    input: image and descriptor
    
    """
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
        self.kpb,self.desb = self.kp1, self.des1
        self.findHomography = False # homography estimated flag
        self.scalebuf = []
        self.scale = 0
        self.H = np.eye(3,dtype=np.float32)
        self.dH1 = np.eye(3,dtype=np.float32)
        self.dH2 = np.eye(3,dtype=np.float32)
        self.matches = []        
        self.inliers = []        

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
    
    def get_goodmatches(self, img):
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
        return pts1, pts2, count

    def track(self,img):
        pts1, pts2, count = self.get_goodmatches(img)

        self.findHomography = False
        self.show = img
        self.matches.append(count)        

        if count > 4:
            self.H, self.mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,3.0)
            if self.check_mask():
                self.get_rect()
                self.get_scale()
                self.findHomography = True
        
        if self.findHomography:
            self.scalebuf.append(self.scale)
            self.inliers.append(self.inliner)
        else:
            self.scalebuf.append(0)
            self.inliers.append(0)

        cv2.imshow("detected",self.show)
        
    def get_rect(self):
            h,w = self.template.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            self.rect = cv2.perspectiveTransform(pts,self.H)
            # draw lines
            self.show = cv2.polylines(self.show,[np.int32(self.rect)],True,255,3, cv2.LINE_AA)
    
    def check_mask(self):
        self.inliner = np.count_nonzero(self.mask)
        print("inliner : "+str(self.inliner)+" in "+str(len(self.mask)))
        #self.total = self.mask.size
        if self.inliner > len(self.mask)*0.4:
            return 1
        else:
            return 0
    
    def get_scale(self):
        sq = self.H[0:1,0:1]*self.H[0:1,0:1]
        self.scale = math.sqrt(sq.sum()/2)

    def show_scale(self):
        leng = len(self.scalebuf)
        fig = plt.figure()
        plt.plot(np.arange(0,leng,1),np.array(self.scalebuf),label="Scale")
        plt.title("Scaling")
        plt.xlabel("Sequence")
        plt.ylabel("scaling")
        plt.ylim(0,2)
        plt.legend()
        plt.grid()
        fig2 = plt.figure()
        plt.plot(np.arange(0,leng,1),np.array(self.inliers),label="Inlier")
        plt.plot(np.arange(0,leng,1),np.array(self.matches),label="Match")
        plt.legend()
        plt.grid()
        plt.show()

    def refresh(self,img):
        self.track(img)
        self.kpb, self.desb = self.kp1, self.des1

# Minor change
class ContinuousTempTracker(TempTracker):
    """
    Update template when get good matchings
    """
    def ctrack(self,img):
        if len(img.shape) > 2: #if color then convert BGR to GRAY
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #print(len(self.kp1))
        kp2,des2 = self.detector.detectAndCompute(img,None)
        if len(kp2) < 5:
            return
        
        # match with buff image    
        matches = self.bf.knnMatch(self.desb,des2,k=2)
        good = []
        pts1 = []
        pts2 = []
        gdes2 = []
        count = 0
        for m,n in matches:      
            if m.distance < 0.6*n.distance:
                good.append(kp2[m.trainIdx])
                pts2.append(kp2[m.trainIdx].pt)
                gdes2.append(des2[m.trainIdx])
                pts1.append(self.kpb[m.queryIdx].pt)
                count += 1
        pts1_ = np.float32(pts1)
        pts2_ = np.float32(pts2)
        gdes2 = np.array(gdes2)

        self.matches.append(count)               
        self.findHomography = False
        self.show = img

        if count > 4:
            self.dH2, self.mask = cv2.findHomography(pts1_, pts2_, cv2.RANSAC,3.0)
            if self.check_mask():
                self.H = np.dot(self.dH2, self.H)
                self.dH = np.dot(self.dH2, self.dH1)
                self.get_rect()
                self.get_scale()
                self.findHomography = True
                self.getnewtemp(img)
        if self.findHomography:
            self.scalebuf.append(self.scale)
            self.inliers.append(self.inliner)
        else:
            self.scalebuf.append(0)
            self.inliers.append(0)

        cv2.imshow("detected",self.show)
        
    def getnewtemp(self,img):
        hei, wid = self.show.shape
        ymin = max(math.floor(self.rect[:,0,1].min()),0)
        ymax = min(math.floor(self.rect[:,0,1].max()),hei-1)
        xmin = max(math.floor(self.rect[:,0,0].min()),0)
        xmax = min(math.floor(self.rect[:,0,0].max()),wid-1)
        temp = img[ymin:ymax,xmin:xmax]
        self.dH1 = np.eye(3,dtype=np.float32)
        self.dH1[0,2]=-xmin
        self.dH1[1,2]=-ymin
        self.H = np.dot(self.dH1,self.H)
        self.kpb, self.desb = self.detector.detectAndCompute(temp,None) 
        cv2.imshow("template",temp)


## main function for parser
import argparse
def load_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-f','--file', help='input video file or video port',default=0) 
    parser.add_argument('-t','--template', help='template filename', default="object.png")  
    parser.add_argument('-d','--descriptor', help='feature descripter', default="ORB")  

    args = parser.parse_args()
    
    # catch the input like "1"
    vfile = int(args.file) if args.file.isdigit() else args.file

    return vfile, args.template, args.descriptor

## Main Function
if __name__ == '__main__' :
    print("Opencv Version is...")
    print(cv2.__version__)
    

    vfile, template, DES = load_args()
    print("Using "+DES+" Detector! \n")

    # video reader
    video = cv2.VideoCapture(vfile)
 
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video!")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()
    
    # read template: enable to read files with 2bytes chalactors 
    temp = imread(template)
    exit("can not open template!") if temp is None else cv2.imshow("template",temp)
    
    tracker = TempTracker(temp,DES)
    T = Checktime()

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        
        # Tracking Object
        tracker.track(frame)
        #T.check()
        
        # Exit if "Q" pressed
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') :
            T.show()
            tracker.show_scale()
            break
        if k == ord('s') :
            cv2.imwrite('result.png',tracker.show)    
            break
        if k == ord('r') :
            tracker.refresh(frame)
