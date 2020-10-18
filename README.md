# R-TOD: Real-Time Object Detector with Minimized End-to-End Delay for Autonomous Driving
### Hardware
* Nvidia Jetson AGX Xavier
* Logitech C930e USB camera
### Software
* Ubuntu 18.04 with JetPack-4.2.2
* OpenCV-3.3.1
* CUDA 10.0

# More details
* http://pjreddie.com/darknet/yolo/
* https://github.com/AlexeyAB/darknet

# Installation 
* Clone R-TOD (Submodule: https://github.com/AveesLab/OpenCV-3.3.1)
```
$ git clone --recursive https://github.com/AveesLab/R-TOD
```
* To use **On-demand Capture** with OpenCV, you don't need any modification. Just build it. See **OpenCV rebuild** in https://github.com/AveesLab/OpenCV-3.3.1.

# Compile using 'Make' 
* `V4L2=1`: Fetch image with On-demand capture method using V4L2 ioctl without OpenCV library (0: Fetch image using OpenCV).
* `ZERO_SLACK=1`: Use Zero-Slack Pipeline method
* `CONTENTION_FREE=1`: Use Contention-Free Pipeline method
* `MEASUREMENT=1`: Measure delay (capture ~ display) and log to csv file (See [Measurement setup](#measurement-setup)).

# Measurement setup 
* If you build with `MEASUREMENT=0`, application will not stop until terminated by user.
* In `src/rtod.h`, you can modify measurement setup.
```
/* Measurement */
#define MEASUREMENT_PATH      // Directory of measurement file
#define MEASUREMENT_FILE      // Measurement file name
#define OBJ_DET_CYCLE_IDX     // Count of measurement
```

# Usage 

### Original Darknet
```
$ ./darknet detector demo cfg/coco.data cfg weights 
      cfg: YOLO network configure file
  weights: weights file
```
### +On-demand Capture
* If you build with `V4L2=0`: See **Capture methods** in https://github.com/AveesLab/OpenCV-3.3.1.
* If you build with `V4L2=1`: No setup required.
```
$ ./darknet detector demo cfg/coco.data cfg weights 
      cfg: YOLO network configure file
  weights: weights file
```
### Zero-Slack Pipeline
* **Zero-Slack Pipeline** needs **On-demand Capture**. 
* Build with `ZERO_SLACK=1`.
```
$ ./darknet detector rtod cfg/coco.data cfg weights
       cfg: YOLO network configure file
   weights: weights file
```
### Contention-Free Pipeline
* **Contention-Free Pipeline** needs **On-demand Capture**. 
* Build with `CONTENTION_FREE=1`.
```
$ ./darknet detector rtod cfg/coco.data cfg weights
       cfg: YOLO network configure file
   weights: weights file
```

# Citation
The paper can be found [here](https://arxiv.org/pdf/2011.06372.pdf). For citation, please use the following Bibtex.
```
@misc{jang2020rtod,
      title={R-TOD: Real-Time Object Detector with Minimized End-to-End Delay for Autonomous Driving}, 
      author={Wonseok Jang and Hansaem Jeong and Kyungtae Kang and Nikil Dutt and Jong-Chan Kim},
      year={2020},
      eprint={2011.06372},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
