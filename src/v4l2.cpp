#ifdef V4L2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <libv4l2.h>

#include "darknet.h"
#include "image_opencv.h"
#include "v4l2_opencv.h"

#define DEFAULT_V4L_QLEN 1 /* Queue length */
#define DEFAULT_CAM_FPS 30 /* 30 FPS */
#define DEFAULT_FRAME_WIDTH 640 
#define DEFAULT_FRAME_HEIGHT 480

//#define DEBUG
//#define BGR24
//#define MJPEG 
#define YUYV 

struct timeval tv = {0};
struct v4l2_buffer buf;

using namespace cv;

/* On demand capture flag*/
int on_demand;

struct buffer {
    void *start;
    size_t length;
};

buffer *buffers;

extern "C" {
    /* Convert IplImage to darknet image */
    image iplImg_to_image(IplImage* src)
    {
        int h = src->height;
        int w = src->width;
        int c = src->nChannels;
        image im = make_image(w, h, c);
        unsigned char *data = (unsigned char *)src->imageData;
        int step = src->widthStep;
        int i, j, k;

        for(i = 0; i < h; ++i){
            for(k= 0; k < c; ++k){
                for(j = 0; j < w; ++j){
                    im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
                }
            }
        }
        return im;
    }

    /* Convert cv::Mat to darknet image */
    image matImg_to_image(cv::Mat m)
    {
        IplImage ipl = m;
        image im = iplImg_to_image(&ipl);
        rgbgr_image(im);
        return im;
    }

    /* Wrapper function */
    static int xioctl(int fd, int request, void *arg)
    {
        int r;

        do r = ioctl (fd, request, arg);
        while (-1 == r && EINTR == errno);

        return r;
    }

    /* Print camera capability */
    int print_caps(int fd, int w, int h)
    {
        struct v4l2_capability caps = {};
        if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &caps))
        {
            perror("Querying Capabilities");
            return -1;
        }

        printf( "Driver Caps:\n"
                "  Driver: \"%s\"\n"
                "  Card: \"%s\"\n"
                "  Bus: \"%s\"\n"
                "  Version: %d.%d\n"
                "  Capabilities: %08x\n",
                caps.driver,
                caps.card,
                caps.bus_info,
                (caps.version>>16)&&0xff,
                (caps.version>>24)&&0xff,
                caps.capabilities);

        struct v4l2_cropcap cropcap = {0};
        cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (-1 == xioctl (fd, VIDIOC_CROPCAP, &cropcap))
        {
            perror("Querying Cropping Capabilities");
            return -1;
        }

        printf( "Camera Cropping:\n"
                "  Bounds: %dx%d+%d+%d\n"
                "  Default: %dx%d+%d+%d\n"
                "  Aspect: %d/%d\n",
                cropcap.bounds.width, cropcap.bounds.height, cropcap.bounds.left, cropcap.bounds.top,
                cropcap.defrect.width, cropcap.defrect.height, cropcap.defrect.left, cropcap.defrect.top,
                cropcap.pixelaspect.numerator, cropcap.pixelaspect.denominator);

        int support_grbg10 = 0;

        struct v4l2_fmtdesc fmtdesc = {0};
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        char fourcc[5] = {0};
        char c, e;
        printf("  FMT : CE Desc\n--------------------\n");
        while (0 == xioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc))
        {
            strncpy(fourcc, (char *)&fmtdesc.pixelformat, 4);
            if (fmtdesc.pixelformat == V4L2_PIX_FMT_SGRBG10)
                support_grbg10 = 1;
            c = fmtdesc.flags & 1? 'C' : ' ';
            e = fmtdesc.flags & 2? 'E' : ' ';
            printf("  %s: %c%c %s\n", fourcc, c, e, fmtdesc.description);
            fmtdesc.index++;
        }

        if(!w) w = DEFAULT_FRAME_WIDTH;
        if(!h) h = DEFAULT_FRAME_HEIGHT;

        struct v4l2_format fmt = {0};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = w;
        fmt.fmt.pix.height = h;
        //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24;
        //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_GREY;
        //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24;
#if (defined MJEPG)
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
#elif (defined YUYV)
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
#endif
        fmt.fmt.pix.field = V4L2_FIELD_NONE;

        if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
        {
            perror("Setting Pixel Format");
            return -1;
        }

        strncpy(fourcc, (char *)&fmt.fmt.pix.pixelformat, 4);
        printf( "Selected Camera Mode:\n"
                "  Width: %d\n"
                "  Height: %d\n"
                "  PixFmt: %s\n"
                "  Field: %d\n",
                fmt.fmt.pix.width,
                fmt.fmt.pix.height,
                fourcc,
                fmt.fmt.pix.field);
        return 1;
    }

    /* Memory mapping */
    int init_mmap(int fd)
    {
        struct v4l2_requestbuffers req = {0};
        int ret;

        req.count = DEFAULT_V4L_QLEN;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req))
        {
            perror("Requesting Buffer");
            return -1;
        }

        //buffers = (buffer*) calloc(req.count, sizeof(*buffers));
        buffers = (struct buffer *) malloc(req.count * sizeof(struct buffer));

        if(!buffers)
        {
            perror("Out of memory");
            return -1;
        }

        for (int i = 0; i < req.count; i++) {
            memset(&buf, 0x00, sizeof(struct v4l2_buffer));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;

            ret = xioctl(fd, VIDIOC_QUERYBUF, &buf);
			if (ret < 0) {
				perror("QUERYBUF");
				return -1;
			}
			buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
            buffers[i].length = buf.length;
        }

        return 1;
    }

    /* Set camera FPS */
    int set_framerate(int fd, int fps)
    {
        struct v4l2_streamparm parm;
        double frame_rate;

        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        if(!fps) fps = DEFAULT_CAM_FPS;

        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = fps;

        if(xioctl(fd, VIDIOC_S_PARM, &parm) < 0) {  
            fprintf(stderr, "VIDEOIO ERROR: V4L: Unable to set %d fps\n", fps);
            return -1;
        }

        frame_rate = (double) parm.parm.capture.timeperframe.denominator / (double)  parm.parm.capture.timeperframe.numerator ;

        printf("  Camera FPS: %0.0f\n", frame_rate);

        return 1;
    }

    /* Capture */
    int capture_image(struct frame_data *f, int fd)
    {
        enum v4l2_buf_type type;

        memset(&buf, 0x00, sizeof(struct v4l2_buffer));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if(-1 == xioctl(fd, VIDIOC_STREAMON, &buf.type))
        {
            perror("Start Capture");
			return -1;
        }

        /* On demand capture */
        if(-1 == xioctl(fd, VIDIOC_QBUF, &buf))
        {
            perror("Query Buffer");
			return -1;
        }

		fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd, &fds);
        tv.tv_sec = 2;
        tv.tv_usec = 0;

        double select_start = get_time_in_ms();

        if(-1 == select(fd + 1, &fds, NULL, NULL, &tv))
		{
			perror("Select");
			return -1;
		}

        f->select = get_time_in_ms() - select_start;

        if(-1 == xioctl(fd, VIDIOC_DQBUF, &buf))
        {
            perror("Retrieving Frame");
			return -1;
        }

        /* Load frame data */
        f->frame_timestamp = (double)buf.timestamp.tv_sec*1000 
            + (double)buf.timestamp.tv_usec*0.001;

        f->frame_sequence = buf.sequence;

        f->length = buf.length;

//    printf("got data in buff %d, len=%d, flags=0x%X, seq=%d, used=%d)\n",
//            buf.index, buf.length, buf.flags, buf.sequence, buf.bytesused);

#ifdef DEBUG
        printf("got data in buff %d, len=%d, flags=0x%X, seq=%d, used=%d)\n",
                buf.index, buf.length, buf.flags, buf.sequence, buf.bytesused);

        printf("image capture time : %f\n", buf.timestamp.tv_sec*1000+(double)buf.timestamp.tv_usec*0.001);
        printf("select time : %f\n", select_time);
        printf("frame timestamp : %f\n", f->frame_timestamp);
        printf("frame sequence : %d\n", f->frame_sequence);
#endif
        image im;

#if (defined MJPEG)
        IplImage* frame;

        /* convert v4l2 raw image to Mat image */
        CvMat cvmat = cvMat(480, 640, CV_8UC3, buffers[buf.index].start);
        frame = cvDecodeImage(&cvmat, 1);

        /* convert IplImage to darknet image type */
        im = iplImg_to_image(frame);
        rgbgr_image(im);
#elif (defined YUYV)
        cv::Mat yuyv_frame, preview;

        /* convert v4l2 raw image to Mat image */
        yuyv_frame = cv::Mat(480, 640, CV_8UC2, buffers[buf.index].start);

        cv::cvtColor(yuyv_frame, preview, COLOR_YUV2BGR_YUY2);

        im = matImg_to_image(preview);
#endif
		f->frame = im;

        return 1; /* return Image as darknet image type */
    }

    /* Check Q size defined in environment variable */
    int get_v4l2_Q_size()
    {
        int env_var_int;
        char *env_var;
        static int size;

        env_var = getenv("V4L2_QLEN");

        if(env_var != NULL){
            env_var_int = atoi(env_var);
        }
        else {
            printf("Using DEFAULT V4L Queue Length\n");
            env_var_int = DEFAULT_V4L_QLEN;
        }

        switch(env_var_int){
            case 0 :
                on_demand = 1;
                size = 1;
                break;
            case 1:
                on_demand = 0;
                size = 1;
                break;
            case 2:
                on_demand = 0;
                size = 2;
                break;
            case 3:
                on_demand = 0;
                size = 3;
                break;
            case 4:
                on_demand = 0;
                size = 4;
                break;
            default :
                on_demand = 0;
                size = 4;
        }
        //printf("%d %d\n", on_demand, size);

        return size;
    }

    /* Open camera device */
    int *open_device(char *cam_dev, int fps, int w, int h)
    {
		static int fd;

        fd = open(cam_dev, O_RDWR | O_NONBLOCK, 0);
        if (fd == -1)
        {
            fprintf(stderr, "VIDEOIO ERROR : Opening video device");
            //return -1;
            return NULL;
        }

        print_caps(fd, w, h);

        if (set_framerate(fd, fps) < 0)
        {
            fprintf(stderr, "VIDEOIO ERROR : Unable to set camera FPS\n");
            //return -1;
            return NULL;
        }

        if(init_mmap(fd) == -1)
        {
            fprintf(stderr, "VIDEOIO ERROR : Fail memory mapping");
            //return -1;
            return NULL;
        }

        //return 1;
		return &fd;
    }
}

#endif
