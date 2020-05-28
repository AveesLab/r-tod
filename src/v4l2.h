#ifndef V4L2_H
#define V4L2_H

#include "darknet.h"

image iplImg_to_image(IplImage* src)
int open_device(char *cam_dev, int fps, int w, int h);
static int xioctl(int fd, int request, void *arg);
int print_caps(int fd, int w, int h);
int init_mmap(int fd, int q_len);
//image capture_image(double *frame_timestamp, int buff_index, int *frame_sequence, double *p_image, int *len);
int set_framerate(int fd, int fps);

#endif

//CvMat *capture_image(struct frame_data *f);
//Mat *capture_image(struct frame_data *f);
