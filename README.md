# Image Template Tracker
Template Tracking using Opencv and Python (C++)

Worked on Laptop with 

- Ubuntu 16.04 LTS 
- python2.7 
- opencv 3.2.0

## Requirement
Python package for opencv and opencv-contrib is needed.

To install dependent package, try

```
pip install opencv-python opencv-contrib-python
```


# Description
Given your template in `template.png` file and tracking this planer template using Homography transformation estimation with various types of key point descriptors.

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

- Template image(in white box).
<img src="https://github.com/YoshiRi/TemplateTracker/blob/master/results/result.png" width="500">

- Tracked result looks like:
![demo](https://github.com/YoshiRi/TemplateTracker/blob/master/results/demovideo.gif)

- Calculation time
 

<img src="https://github.com/YoshiRi/TemplateTracker/blob/master/results/siftcalc.png" width="400">

 
 In my laptop with corei5 processor, it took about 100ms to calculation SIFT transformation. 