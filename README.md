# Image Template Tracker
Template Tracking using Opencv and Python (C++)

Worked on Laptop with 

- Windows 10 Home
  - python3.6
  - opencv 3.4.0


## Requirement

- numpy, pandas, matplotlib
- opencv 
  - Opencv package above 3.4.0 and below 4.x (Due to the feature extraction method.)
  - both opencv and opencv-contrib is needed.



# How to use


## python version


```
python TempTracking.py -d <keypoint descriptor> -f <input video file> -t <template image>
```

- `<keypoint descriptor>` support `ORB, SIFT, SURF, AKAZE, KAZE` feature. 
- `<input video files>` need to be in the ascii coded folder 
- `<template image>` is the template image you want to track
- Default values are `ORB`, video 0 and  `object.png`

With the program in `ScalingAndTimeMeasurement` branch, it will show the calculation time and estimated scales.

## c++ (opencv with cuda8.0) version

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

- Calculation time ~ 100ms with SIFT in core i5-3000 laptop
  

<img src="https://github.com/YoshiRi/TemplateTracker/blob/master/results/siftcalc.png" width="400">
