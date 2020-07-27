# Yolo-v4 and Yolo-v3/v2 for Windows and Linux
### (neural network for object detection) - Tensor Cores can be used on [Linux](https://github.com/AlexeyAB/darknet#how-to-compile-on-linux) and [Windows](https://github.com/AlexeyAB/darknet#how-to-compile-on-windows-using-cmake-gui)

More details
* http://pjreddie.com/darknet/yolo/
* https://github.com/AlexeyAB/darknet

# R-TOD: Real-Time Object Detector

### Compile using 'Make' ###
* `V4L2=1`: Fetch image with On-demand capture using V4L2 ioctl without OpenCV library (0: Fetch image using OpenCV).
* `ZERO_SLACK=1`: Use Zero-slack pipeline (0 means Contention-free pipeline).
* `MEASUREMENT=1`: Measure delay (capture ~ display) and log to csv file (See [Measurement setup](#measurement-setup)).

### How to set On-demand capture
* Build `V4L2=0`: See https://github.com/wonseok-Jang/OpenCV-3.3.1.
* Build `V4L2=1`: No setup required.

### Measurement setup ###
* If you build with `MEASUREMENT=0`, never stop until terminated by user.
* In `src/rtod.h`, you can modify measurement setup.
```
/* Measurement */
#define MEASUREMENT_PATH               // Directory of measurement file
#define MEASUREMENT_FILE               // Measurement file name
#define MEASUREMENT_OBJ_DET_CYCLE_IDX  // Count of measurement
```

### Usage ###

#### Original pipeline
You can choose two capture method (Orignal capture & On-demand capture).
* See **Image capture** in https://github.com/wonseok-Jang/OpenCV-3.3.1.
* **Original capture**: Orignal darknet with nothing modified.
* **On-demand capture**: Remove unnecessary image queue.
```
$ ./darknet detector demo cfg/coco.data cfg weights 
     cfg : path to yolo network configure file
  weights: path to weights file
```
#### Zero-slack pipeline
* See [How to set On-demand capture](#how-to-set-on--demand-capture).
* Compile with `ZERO_SLACK=1`.
```
$ ./darknet detector rtod cfg/coco.data cfg weights
      cfg : path to yolo network configure file
      weights: path to weights file
```
#### Contention-free pipeline
* See [How to set On-demand capture](#how-to-set-on--demand-capture).
* Compile with `ZERO_SLACK=0`.
```
$ ./darknet detector rtod cfg/coco.data cfg weights
      cfg : path to yolo network configure file
   weights: path to weights file
```
