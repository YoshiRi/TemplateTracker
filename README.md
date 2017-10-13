# Image Template Tracker
Template Tracking using Opencv and Python (C++)

Worked on Ubuntu 16.04 LTS python2.7 opencv 3.2.0

# Description
Given your template in "template.png" file and tracking this planer template using Homography transformation estimation with various types of key point descriptors.

# How to use
## python version
```
python ObjTracking.py <keypoint descriptor>
```
\<keypoint descriptor\> can be ORB, SIFT, SURF, AKAZE, KAZE feature methods.

When you stopped program, you can see the calclation time.

## opencv cuda version
Do following to compile and just run the program.
```
g++ -ggdb TempTrackGPU.cpp `pkg-config --cflags --libs opencv`
```
It only support SURF with cuda8.0, and you need opencv3 compiled with cuda.

# DEMO
Sample Result.

<img src="https://github.com/YoshiRi/TemplateTracker/blob/master/results/result.png" width="500">

In my case, it tooks about 100ms to calcurate SIFT transformation. 

<img src="https://github.com/YoshiRi/TemplateTracker/blob/master/results/siftcalc.png" width="400">
