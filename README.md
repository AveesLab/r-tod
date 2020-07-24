# Yolo-v4 and Yolo-v3/v2 for Windows and Linux
### (neural network for object detection) - Tensor Cores can be used on [Linux](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux) and [Windows](https://github.com/AlexeyAB/darknet#how-to-compile-on-windows-using-cmake-gui)

More details: http://pjreddie.com/darknet/yolo/
              https://github.com/AlexeyAB/darknet

# R-TOD: Real-Time Object Detector

### Compile using 'Make' ###
* `V4L2=1` build with V4L2 - Fetch image with On-demand capture using V4L2 ioctl without OpenCV library
* `ZERO_SLACK=1` build with ZERO_SLACK - Zero-slack pipeline (0 means Contention-free pipeline)
* `MEASUREMENT=1` build with MEASUREMENT - Measure delay (capture ~ display) and log to csv file (You can define iteration in src/rtod.h OBJ_DET_CYCLE_IDX)

### Image fetch with OpenCV
See https://github.com/wonseok-Jang/OpenCV-3.3.1

## Usage ###
```
# Original capture & Original pipeline
* $ ./darknet detector demo cfg/coco.data cfg weights
     cfg : path to yolo network configure file
  weights: path to weights file
  
# On-demand capture & Original pipeline
* Set On-demand capture
   case 1: build with V4L2=0
         - See Image fetch with OpenCV
   case 2: build with V4L2=1
         - No setup required
* ./darknet detector demo cfg/coco.data cfg weights
      cfg : path to yolo network configure file
   weights: path to weights file
  
# Zero-slack pipeline
* Set On-demand capture
* Compile with ZERO_SLACK=1
* ./darknet detector rtod cfg/coco.data cfg weights
      cfg : path to yolo network configure file
   weights: path to weights file

# Contention-free pipeline
* Set On-demand capture
* Compile with ZERO_SLACK=0
* ./darknet detector rtod cfg/coco.data cfg weights
      cfg : path to yolo network configure file
   weights: path to weights file
'''
