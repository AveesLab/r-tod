#ifdef V4L2

#ifndef V4L2_H
#define V4L2_H

#include "darknet.h"
#include "image_opencv.h"

int *open_device(char *cam_dev, int fps, int w, int h);
static int xioctl(int fd, int request, void *arg);
int print_caps(int fd, int w, int h);
int init_mmap(int fd);
int capture_image(struct frame_data *f, int fd);
int set_framerate(int fd, int fps);

#endif

#endif
