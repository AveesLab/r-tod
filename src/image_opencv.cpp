#include "image_opencv.h"

#ifdef OPENCV
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <opencv2/core/version.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

// includes for OpenCV >= 3.x
#ifndef CV_VERSION_EPOCH
#include <opencv2/core/types.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#endif

// OpenCV includes for OpenCV 2.x
#ifdef CV_VERSION_EPOCH
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/core/version.hpp>
#endif

//using namespace cv;

using std::cerr;
using std::endl;

#ifdef DEBUG
#define OCV_D "d"
#else
#define OCV_D
#endif//DEBUG


// OpenCV libraries
#ifndef CV_VERSION_EPOCH
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION) OCV_D
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#else   // CV_VERSION_EPOCH
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR) OCV_D
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#endif    // CV_VERSION_EPOCH

#include "http_stream.h"

#ifndef CV_RGB
#define CV_RGB(r, g, b) cvScalar( (b), (g), (r), 0 )
#endif

#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif

#ifndef CV_AA
#define CV_AA cv::LINE_AA
#endif

extern "C" {

    struct mat_cv : cv::Mat { int a[0]; };
    struct cap_cv : cv::VideoCapture { int a[0]; };
    struct write_cv : cv::VideoWriter { int a[0]; };

// ====================================================================
// cv::Mat
// ====================================================================
    image mat_to_image(cv::Mat mat);
    cv::Mat image_to_mat(image img);
//    image ipl_to_image(mat_cv* src);
//    mat_cv *image_to_ipl(image img);
//    cv::Mat ipl_to_mat(IplImage *ipl);
//    IplImage *mat_to_ipl(cv::Mat mat);


mat_cv *load_image_mat_cv(const char *filename, int flag)
{
    try {
        cv::Mat *mat_ptr = new cv::Mat();
        cv::Mat &mat = *mat_ptr;
        mat = cv::imread(filename, flag);
        if (mat.empty())
        {
            delete mat_ptr;
            std::string shrinked_filename = filename;
            if (shrinked_filename.length() > 1024) {
                shrinked_filename += "name is too long: ";
                shrinked_filename.resize(1024);
            }
            cerr << "Cannot load image " << shrinked_filename << std::endl;
            std::ofstream bad_list("bad.list", std::ios::out | std::ios::app);
            bad_list << shrinked_filename << std::endl;
            //if (check_mistakes) getchar();
            return NULL;
        }
        if (mat.channels() == 3) cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        else if (mat.channels() == 4) cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);

        return (mat_cv *)mat_ptr;
    }
    catch (...) {
        cerr << "OpenCV exception: load_image_mat_cv \n";
    }
    return NULL;
}
// ----------------------------------------

cv::Mat load_image_mat(char *filename, int channels)
{
    int flag = cv::IMREAD_UNCHANGED;
    if (channels == 0) flag = cv::IMREAD_COLOR;
    else if (channels == 1) flag = cv::IMREAD_GRAYSCALE;
    else if (channels == 3) flag = cv::IMREAD_COLOR;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    //flag |= IMREAD_IGNORE_ORIENTATION;    // un-comment it if you want

    cv::Mat *mat_ptr = (cv::Mat *)load_image_mat_cv(filename, flag);

    if (mat_ptr == NULL) {
        return cv::Mat();
    }
    cv::Mat mat = *mat_ptr;
    delete mat_ptr;

    return mat;
}
// ----------------------------------------

image load_image_cv(char *filename, int channels)
{
    cv::Mat mat = load_image_mat(filename, channels);

    if (mat.empty()) {
        return make_image(10, 10, channels);
    }
    return mat_to_image(mat);
}
// ----------------------------------------

image load_image_resize(char *filename, int w, int h, int c, image *im)
{
    image out;
    try {
        cv::Mat loaded_image = load_image_mat(filename, c);

        *im = mat_to_image(loaded_image);

        cv::Mat resized(h, w, CV_8UC3);
        cv::resize(loaded_image, resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        out = mat_to_image(resized);
    }
    catch (...) {
        cerr << " OpenCV exception: load_image_resize() can't load image %s " << filename << " \n";
        out = make_image(w, h, c);
        *im = make_image(w, h, c);
    }
    return out;
}
// ----------------------------------------

int get_width_mat(mat_cv *mat)
{
    if (mat == NULL) {
        cerr << " Pointer is NULL in get_width_mat() \n";
        return 0;
    }
    return ((cv::Mat *)mat)->cols;
}
// ----------------------------------------

int get_height_mat(mat_cv *mat)
{
    if (mat == NULL) {
        cerr << " Pointer is NULL in get_height_mat() \n";
        return 0;
    }
    return ((cv::Mat *)mat)->rows;
}
// ----------------------------------------

void release_mat(mat_cv **mat)
{
    try {
        cv::Mat **mat_ptr = (cv::Mat **)mat;
        if (*mat_ptr) delete *mat_ptr;
        *mat_ptr = NULL;
    }
    catch (...) {
        cerr << "OpenCV exception: release_mat \n";
    }
}

// ====================================================================
// IplImage
// ====================================================================
/*
int get_width_cv(mat_cv *ipl_src)
{
    IplImage *ipl = (IplImage *)ipl_src;
    return ipl->width;
}
// ----------------------------------------

int get_height_cv(mat_cv *ipl_src)
{
    IplImage *ipl = (IplImage *)ipl_src;
    return ipl->height;
}
// ----------------------------------------

void release_ipl(mat_cv **ipl)
{
    IplImage **ipl_img = (IplImage **)ipl;
    if (*ipl_img) cvReleaseImage(ipl_img);
    *ipl_img = NULL;
}
// ----------------------------------------

// ====================================================================
// image-to-ipl, ipl-to-image, image_to_mat, mat_to_image
// ====================================================================

mat_cv *image_to_ipl(image im)
{
    int x, y, c;
    IplImage *disp = cvCreateImage(cvSize(im.w, im.h), IPL_DEPTH_8U, im.c);
    int step = disp->widthStep;
    for (y = 0; y < im.h; ++y) {
        for (x = 0; x < im.w; ++x) {
            for (c = 0; c < im.c; ++c) {
                float val = im.data[c*im.h*im.w + y*im.w + x];
                disp->imageData[y*step + x*im.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return (mat_cv *)disp;
}
// ----------------------------------------

image ipl_to_image(mat_cv* src_ptr)
{
    IplImage* src = (IplImage*)src_ptr;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)src->imageData;
    int step = src->widthStep;
    int i, j, k;

    for (i = 0; i < h; ++i) {
        for (k = 0; k < c; ++k) {
            for (j = 0; j < w; ++j) {
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k] / 255.;
            }
        }
    }
    return im;
}
// ----------------------------------------

cv::Mat ipl_to_mat(IplImage *ipl)
{
    Mat m = cvarrToMat(ipl, true);
    return m;
}
// ----------------------------------------

IplImage *mat_to_ipl(cv::Mat mat)
{
    IplImage *ipl = new IplImage;
    *ipl = mat;
    return ipl;
}
// ----------------------------------------
*/

cv::Mat image_to_mat(image img)
{
    int channels = img.c;
    int width = img.w;
    int height = img.h;
    cv::Mat mat = cv::Mat(height, width, CV_8UC(channels));
    int step = mat.step;

    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            for (int c = 0; c < img.c; ++c) {
                float val = img.data[c*img.h*img.w + y*img.w + x];
                mat.data[y*step + x*img.c + c] = (unsigned char)(val * 255);
            }
        }
    }
    return mat;
}
// ----------------------------------------

image mat_to_image(cv::Mat mat)
{
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)mat.data;
    int step = mat.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                //uint8_t val = mat.ptr<uint8_t>(y)[c * x + k];
                //uint8_t val = mat.at<Vec3b>(y, x).val[k];
                //im.data[k*w*h + y*w + x] = val / 255.0f;

                im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
            }
        }
    }
    return im;
}

image mat_to_image_cv(mat_cv *mat)
{
    return mat_to_image(*mat);
}

// ====================================================================
// Window
// ====================================================================
void create_window_cv(char const* window_name, int full_screen, int width, int height)
{
    try {
        int window_type = cv::WINDOW_NORMAL;
#ifdef CV_VERSION_EPOCH // OpenCV 2.x
        if (full_screen) window_type = CV_WINDOW_FULLSCREEN;
#else
        if (full_screen) window_type = cv::WINDOW_FULLSCREEN;
#endif
        cv::namedWindow(window_name, window_type);
        cv::moveWindow(window_name, 0, 0);
        cv::resizeWindow(window_name, width, height);
    }
    catch (...) {
        cerr << "OpenCV exception: create_window_cv \n";
    }
}
// ----------------------------------------

void destroy_all_windows_cv()
{
    try {
        cv::destroyAllWindows();
    }
    catch (...) {
        cerr << "OpenCV exception: destroy_all_windows_cv \n";
    }
}
// ----------------------------------------

int wait_key_cv(int delay)
{
    try {
        return cv::waitKey(delay);
    }
    catch (...) {
        cerr << "OpenCV exception: wait_key_cv \n";
    }
    return -1;
}
// ----------------------------------------

int wait_until_press_key_cv()
{
    return wait_key_cv(0);
}
// ----------------------------------------

void make_window(char *name, int w, int h, int fullscreen)
{
    try {
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        if (fullscreen) {
#ifdef CV_VERSION_EPOCH // OpenCV 2.x
            cv::setWindowProperty(name, cv::WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
#else
            cv::setWindowProperty(name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
#endif
        }
        else {
            cv::resizeWindow(name, w, h);
            if (strcmp(name, "Demo") == 0) cv::moveWindow(name, 0, 0);
        }
    }
    catch (...) {
        cerr << "OpenCV exception: make_window \n";
    }
}
// ----------------------------------------

static float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}
// ----------------------------------------

void show_image_cv(image p, const char *name)
{
    try {
        image copy = copy_image(p);
        constrain_image(copy);

        cv::Mat mat = image_to_mat(copy);
        if (mat.channels() == 3) cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        else if (mat.channels() == 4) cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::imshow(name, mat);
        free_image(copy);
    }
    catch (...) {
        cerr << "OpenCV exception: show_image_cv \n";
    }
}
// ----------------------------------------

/*
void show_image_cv_ipl(mat_cv *disp, const char *name)
{
    if (disp == NULL) return;
    char buff[256];
    sprintf(buff, "%s", name);
    cv::namedWindow(buff, WINDOW_NORMAL);
    cvShowImage(buff, disp);
}
// ----------------------------------------
*/

void show_image_mat(mat_cv *mat_ptr, const char *name)
{
    try {
        if (mat_ptr == NULL) return;
        cv::Mat &mat = *(cv::Mat *)mat_ptr;
        cv::namedWindow(name, cv::WINDOW_NORMAL);
        cv::imshow(name, mat);
    }
    catch (...) {
        cerr << "OpenCV exception: show_image_mat \n";
    }
}

// ====================================================================
// Video Writer
// ====================================================================
write_cv *create_video_writer(char *out_filename, char c1, char c2, char c3, char c4, int fps, int width, int height, int is_color)
{
    try {
    cv::VideoWriter * output_video_writer =
#ifdef CV_VERSION_EPOCH
        new cv::VideoWriter(out_filename, CV_FOURCC(c1, c2, c3, c4), fps, cv::Size(width, height), is_color);
#else
        new cv::VideoWriter(out_filename, cv::VideoWriter::fourcc(c1, c2, c3, c4), fps, cv::Size(width, height), is_color);
#endif

    return (write_cv *)output_video_writer;
    }
    catch (...) {
        cerr << "OpenCV exception: create_video_writer \n";
    }
    return NULL;
}

void write_frame_cv(write_cv *output_video_writer, mat_cv *mat)
{
    try {
        cv::VideoWriter *out = (cv::VideoWriter *)output_video_writer;
        out->write(*mat);
    }
    catch (...) {
        cerr << "OpenCV exception: write_frame_cv \n";
    }
}

void release_video_writer(write_cv **output_video_writer)
{
    try {
        if (output_video_writer) {
            std::cout << " closing...";
            cv::VideoWriter *out = *(cv::VideoWriter **)output_video_writer;
            out->release();
            delete out;
            output_video_writer = NULL;
            std::cout << " closed!";
        }
        else {
            cerr << "OpenCV exception: output_video_writer isn't created \n";
        }
    }
    catch (...) {
        cerr << "OpenCV exception: release_video_writer \n";
    }
}

/*
void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return (void *) cap;
}


image get_image_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()) return make_empty_image(0,0,0);
    return mat_to_image(m);
}

int show_image_cv(image im, const char* name, int ms)
{
    Mat m = image_to_mat(im);
    imshow(name, m);
    int c = waitKey(ms);
    if (c != -1) c = c%256;
    return c;
}
*/


// ====================================================================
// Video Capture
// ====================================================================

cap_cv* get_capture_video_stream(const char *path) {
    cv::VideoCapture* cap = NULL;
    try {
        cap = new cv::VideoCapture(path);
    }
    catch (...) {
        cerr << " OpenCV exception: video-stream " << path << " can't be opened! \n";
    }
    return (cap_cv*)cap;
}
// ----------------------------------------

cap_cv* get_capture_webcam(int index)
{
    cv::VideoCapture* cap = NULL;
    try {
        cap = new cv::VideoCapture(index);
        //cap->set(CV_CAP_PROP_FRAME_WIDTH, 1280);
        //cap->set(CV_CAP_PROP_FRAME_HEIGHT, 960);
    }
    catch (...) {
        cerr << " OpenCV exception: Web-camera " << index << " can't be opened! \n";
    }
    return (cap_cv*)cap;
}
// ----------------------------------------

void release_capture(cap_cv* cap)
{
    try {
        cv::VideoCapture *cpp_cap = (cv::VideoCapture *)cap;
        delete cpp_cap;
    }
    catch (...) {
        cerr << " OpenCV exception: cv::VideoCapture " << cap << " can't be released! \n";
    }
}
// ----------------------------------------

mat_cv* get_capture_frame_cv(cap_cv *cap) {
    cv::Mat *mat = new cv::Mat();
    try {
        if (cap) {
            cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
            if (cpp_cap.isOpened())
            {
                cpp_cap >> *mat;
            }
            else std::cout << " Video-stream stopped! \n";
        }
        else cerr << " cv::VideoCapture isn't created \n";
    }
    catch (...) {
        std::cout << " OpenCV exception: Video-stream stoped! \n";
    }
    return (mat_cv *)mat;
}
// ----------------------------------------

int get_stream_fps_cpp_cv(cap_cv *cap)
{
    int fps = 25;
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
        fps = cpp_cap.get(cv::CAP_PROP_FPS);
#else                        // OpenCV 2.x
        fps = cpp_cap.get(CV_CAP_PROP_FPS);
#endif
    }
    catch (...) {
        cerr << " Can't get FPS of source videofile. For output video FPS = 25 by default. \n";
    }
    return fps;
}
// ----------------------------------------

double get_capture_property_cv(cap_cv *cap, int property_id)
{
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
        return cpp_cap.get(property_id);
    }
    catch (...) {
        cerr << " OpenCV exception: Can't get property of source video-stream. \n";
    }
    return 0;
}
// ----------------------------------------

double get_capture_frame_count_cv(cap_cv *cap)
{
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
        return cpp_cap.get(cv::CAP_PROP_FRAME_COUNT);
#else                        // OpenCV 2.x
        return cpp_cap.get(CV_CAP_PROP_FRAME_COUNT);
#endif
    }
    catch (...) {
        cerr << " OpenCV exception: Can't get CAP_PROP_FRAME_COUNT of source videofile. \n";
    }
    return 0;
}
// ----------------------------------------

int set_capture_property_cv(cap_cv *cap, int property_id, double value)
{
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
        return cpp_cap.set(property_id, value);
    }
    catch (...) {
        cerr << " Can't set property of source video-stream. \n";
    }
    return false;
}
// ----------------------------------------

int set_capture_position_frame_cv(cap_cv *cap, int index)
{
    try {
        cv::VideoCapture &cpp_cap = *(cv::VideoCapture *)cap;
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
        return cpp_cap.set(cv::CAP_PROP_POS_FRAMES, index);
#else                        // OpenCV 2.x
        return cpp_cap.set(CV_CAP_PROP_POS_FRAMES, index);
#endif
    }
    catch (...) {
        cerr << " Can't set CAP_PROP_POS_FRAMES of source videofile. \n";
    }
    return false;
}
// ----------------------------------------



// ====================================================================
// ... Video Capture
// ====================================================================

image get_image_from_stream_cpp(cap_cv *cap)
{
    cv::Mat *src = new cv::Mat();
    static int once = 1;
    if (once) {
        once = 0;
        do {
            src = get_capture_frame_cv(cap);
            if (!src) return make_empty_image(0, 0, 0);
        } while (src->cols < 1 || src->rows < 1 || src->channels() < 1);
        printf("Video stream: %d x %d \n", src->cols, src->rows);
    }
    else
        src = get_capture_frame_cv(cap);

    if (!src) return make_empty_image(0, 0, 0);
    image im = mat_to_image(*src);
    rgbgr_image(im);
    return im;
}
// ----------------------------------------

int wait_for_stream(cap_cv *cap, cv::Mat* src, int dont_close)
{
    if (!src) {
        if (dont_close) src = new cv::Mat(416, 416, CV_8UC(3)); // cvCreateImage(cvSize(416, 416), IPL_DEPTH_8U, 3);
        else return 0;
    }
    if (src->cols < 1 || src->rows < 1 || src->channels() < 1) {
        if (dont_close) {
            delete src;// cvReleaseImage(&src);
            int z = 0;
            for (z = 0; z < 20; ++z) {
                get_capture_frame_cv(cap);
                delete src;// cvReleaseImage(&src);
            }
            src = new cv::Mat(416, 416, CV_8UC(3)); // cvCreateImage(cvSize(416, 416), IPL_DEPTH_8U, 3);
        }
        else return 0;
    }
    return 1;
}
// ----------------------------------------

image get_image_from_stream_resize(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close)
{
    c = c ? c : 3;
    cv::Mat *src = NULL;

    static int once = 1;
    if (once) {
        once = 0;
        do {
            src = get_capture_frame_cv(cap);
            if (!src) return make_empty_image(0, 0, 0);
        } while (src->cols < 1 || src->rows < 1 || src->channels() < 1);
        printf("Video stream: %d x %d \n", src->cols, src->rows);
    }
    else
        src = get_capture_frame_cv(cap);

    if (!wait_for_stream(cap, src, dont_close)) return make_empty_image(0, 0, 0);

    *(cv::Mat **)in_img = src;

    cv::Mat new_img = cv::Mat(h, w, CV_8UC(c));
    cv::resize(*src, new_img, new_img.size(), 0, 0, cv::INTER_LINEAR);
    if (c>1) cv::cvtColor(new_img, new_img, cv::COLOR_RGB2BGR);
    image im = mat_to_image(new_img);

    //show_image_cv(im, "im");
    //show_image_mat(*in_img, "in_img");
    return im;
}
// ----------------------------------------

image get_image_from_stream_letterbox(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close)
{
    c = c ? c : 3;
    cv::Mat *src = NULL;
    static int once = 1;
    if (once) {
        once = 0;
        do {
            src = get_capture_frame_cv(cap);
            if (!src) return make_empty_image(0, 0, 0);
        } while (src->cols < 1 || src->rows < 1 || src->channels() < 1);
        printf("Video stream: %d x %d \n", src->cols, src->rows);
    }
    else
        src = get_capture_frame_cv(cap);

    if (!wait_for_stream(cap, src, dont_close)) return make_empty_image(0, 0, 0);   // passes (cv::Mat *)src while should be (cv::Mat **)src

    *in_img = (mat_cv *)new cv::Mat(src->rows, src->cols, CV_8UC(c));
    cv::resize(*src, **in_img, (*in_img)->size(), 0, 0, cv::INTER_LINEAR);

    if (c>1) cv::cvtColor(*src, *src, cv::COLOR_RGB2BGR);
    image tmp = mat_to_image(*src);
    image im = letterbox_image(tmp, w, h);
    free_image(tmp);
    release_mat((mat_cv **)&src);

    //show_image_cv(im, "im");
    //show_image_mat(*in_img, "in_img");
    return im;
}
// ----------------------------------------

// ====================================================================
// Image Saving
// ====================================================================
extern int stbi_write_png(char const *filename, int w, int h, int comp, const void  *data, int stride_in_bytes);
extern int stbi_write_jpg(char const *filename, int x, int y, int comp, const void  *data, int quality);

void save_mat_png(cv::Mat img_src, const char *name)
{
    cv::Mat img_rgb;
    if (img_src.channels() >= 3) cv::cvtColor(img_src, img_rgb, cv::COLOR_RGB2BGR);
    stbi_write_png(name, img_rgb.cols, img_rgb.rows, 3, (char *)img_rgb.data, 0);
}
// ----------------------------------------

void save_mat_jpg(cv::Mat img_src, const char *name)
{
    cv::Mat img_rgb;
    if (img_src.channels() >= 3) cv::cvtColor(img_src, img_rgb, cv::COLOR_RGB2BGR);
    stbi_write_jpg(name, img_rgb.cols, img_rgb.rows, 3, (char *)img_rgb.data, 80);
}
// ----------------------------------------


void save_cv_png(mat_cv *img_src, const char *name)
{
    cv::Mat* img = (cv::Mat* )img_src;
    save_mat_png(*img, name);
}
// ----------------------------------------

void save_cv_jpg(mat_cv *img_src, const char *name)
{
    cv::Mat* img = (cv::Mat*)img_src;
    save_mat_jpg(*img, name);
}
// ----------------------------------------


// ====================================================================
// Draw Detection
// ====================================================================
void draw_detections_cv_v3(mat_cv* mat, detection *dets, int num, float thresh, char **names, image **alphabet, int classes, int ext_output)
{
    try {
        cv::Mat *show_img = mat;
        int i, j;
        if (!show_img) return;
        static int frame_id = 0;
        frame_id++;

        for (i = 0; i < num; ++i) {
            char labelstr[4096] = { 0 };
            int class_id = -1;
            for (j = 0; j < classes; ++j) {
                int show = strncmp(names[j], "dont_show", 9);
                if (dets[i].prob[j] > thresh && show) {
                    if (class_id < 0) {
                        strcat(labelstr, names[j]);
                        class_id = j;
                        char buff[10];
                        sprintf(buff, " (%2.0f%%)", dets[i].prob[j] * 100);
                        strcat(labelstr, buff);
                    }
                    else {
                        strcat(labelstr, ", ");
                        strcat(labelstr, names[j]);
                    }
                    printf("%s: %.0f%% ", names[j], dets[i].prob[j] * 100);
                }
            }
            if (class_id >= 0) {
                int width = std::max(1.0f, show_img->rows * .002f);

                //if(0){
                //width = pow(prob, 1./2.)*10+1;
                //alphabet = 0;
                //}

                //printf("%d %s: %.0f%%\n", i, names[class_id], prob*100);
                int offset = class_id * 123457 % classes;
                float red = get_color(2, offset, classes);
                float green = get_color(1, offset, classes);
                float blue = get_color(0, offset, classes);
                float rgb[3];

                //width = prob*20+2;

                rgb[0] = red;
                rgb[1] = green;
                rgb[2] = blue;
                box b = dets[i].bbox;
                if (std::isnan(b.w) || std::isinf(b.w)) b.w = 0.5;
                if (std::isnan(b.h) || std::isinf(b.h)) b.h = 0.5;
                if (std::isnan(b.x) || std::isinf(b.x)) b.x = 0.5;
                if (std::isnan(b.y) || std::isinf(b.y)) b.y = 0.5;
                b.w = (b.w < 1) ? b.w : 1;
                b.h = (b.h < 1) ? b.h : 1;
                b.x = (b.x < 1) ? b.x : 1;
                b.y = (b.y < 1) ? b.y : 1;
                //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

                int left = (b.x - b.w / 2.)*show_img->cols;
                int right = (b.x + b.w / 2.)*show_img->cols;
                int top = (b.y - b.h / 2.)*show_img->rows;
                int bot = (b.y + b.h / 2.)*show_img->rows;

                if (left < 0) left = 0;
                if (right > show_img->cols - 1) right = show_img->cols - 1;
                if (top < 0) top = 0;
                if (bot > show_img->rows - 1) bot = show_img->rows - 1;

                //int b_x_center = (left + right) / 2;
                //int b_y_center = (top + bot) / 2;
                //int b_width = right - left;
                //int b_height = bot - top;
                //sprintf(labelstr, "%d x %d - w: %d, h: %d", b_x_center, b_y_center, b_width, b_height);

                float const font_size = show_img->rows / 1000.F;
                cv::Size const text_size = cv::getTextSize(labelstr, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, 1, 0);
                cv::Point pt1, pt2, pt_text, pt_text_bg1, pt_text_bg2;
                pt1.x = left;
                pt1.y = top;
                pt2.x = right;
                pt2.y = bot;
                pt_text.x = left;
                pt_text.y = top - 4;// 12;
                pt_text_bg1.x = left;
                pt_text_bg1.y = top - (3 + 18 * font_size);
                pt_text_bg2.x = right;
                if ((right - left) < text_size.width) pt_text_bg2.x = left + text_size.width;
                pt_text_bg2.y = top;
                cv::Scalar color;
                color.val[0] = red * 256;
                color.val[1] = green * 256;
                color.val[2] = blue * 256;

                // you should create directory: result_img
                //static int copied_frame_id = -1;
                //static IplImage* copy_img = NULL;
                //if (copied_frame_id != frame_id) {
                //    copied_frame_id = frame_id;
                //    if(copy_img == NULL) copy_img = cvCreateImage(cvSize(show_img->width, show_img->height), show_img->depth, show_img->nChannels);
                //    cvCopy(show_img, copy_img, 0);
                //}
                //static int img_id = 0;
                //img_id++;
                //char image_name[1024];
                //sprintf(image_name, "result_img/img_%d_%d_%d_%s.jpg", frame_id, img_id, class_id, names[class_id]);
                //CvRect rect = cvRect(pt1.x, pt1.y, pt2.x - pt1.x, pt2.y - pt1.y);
                //cvSetImageROI(copy_img, rect);
                //cvSaveImage(image_name, copy_img, 0);
                //cvResetImageROI(copy_img);

                cv::rectangle(*show_img, pt1, pt2, color, width, 8, 0);
                if (ext_output)
                    printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
                    (float)left, (float)top, b.w*show_img->cols, b.h*show_img->rows);
                else
                    printf("\n");

                cv::rectangle(*show_img, pt_text_bg1, pt_text_bg2, color, width, 8, 0);
                cv::rectangle(*show_img, pt_text_bg1, pt_text_bg2, color, CV_FILLED, 8, 0);    // filled
                cv::Scalar black_color = CV_RGB(0, 0, 0);
                cv::putText(*show_img, labelstr, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, black_color, 2 * font_size, CV_AA);
                // cv::FONT_HERSHEY_COMPLEX_SMALL, cv::FONT_HERSHEY_SIMPLEX
            }
        }
        if (ext_output) {
            fflush(stdout);
        }
    }
    catch (...) {
        cerr << "OpenCV exception: draw_detections_cv_v3() \n";
    }
}
// ----------------------------------------

// ====================================================================
// Draw Loss & Accuracy chart
// ====================================================================
mat_cv* draw_train_chart(float max_img_loss, int max_batches, int number_of_lines, int img_size, int dont_show)
{
    int img_offset = 60;
    int draw_size = img_size - img_offset;
    cv::Mat *img_ptr = new cv::Mat(img_size, img_size, CV_8UC3, CV_RGB(255, 255, 255));
    cv::Mat &img = *img_ptr;
    cv::Point pt1, pt2, pt_text;

    try {
        char char_buff[100];
        int i;
        // vertical lines
        pt1.x = img_offset; pt2.x = img_size, pt_text.x = 30;
        for (i = 1; i <= number_of_lines; ++i) {
            pt1.y = pt2.y = (float)i * draw_size / number_of_lines;
            cv::line(img, pt1, pt2, CV_RGB(224, 224, 224), 1, 8, 0);
            if (i % 10 == 0) {
                sprintf(char_buff, "%2.1f", max_img_loss*(number_of_lines - i) / number_of_lines);
                pt_text.y = pt1.y + 3;

                cv::putText(img, char_buff, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
                cv::line(img, pt1, pt2, CV_RGB(128, 128, 128), 1, 8, 0);
            }
        }
        // horizontal lines
        pt1.y = draw_size; pt2.y = 0, pt_text.y = draw_size + 15;
        for (i = 0; i <= number_of_lines; ++i) {
            pt1.x = pt2.x = img_offset + (float)i * draw_size / number_of_lines;
            cv::line(img, pt1, pt2, CV_RGB(224, 224, 224), 1, 8, 0);
            if (i % 10 == 0) {
                sprintf(char_buff, "%d", max_batches * i / number_of_lines);
                pt_text.x = pt1.x - 20;
                cv::putText(img, char_buff, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
                cv::line(img, pt1, pt2, CV_RGB(128, 128, 128), 1, 8, 0);
            }
        }

        cv::putText(img, "Loss", cv::Point(10, 55), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 255), 1, CV_AA);
        cv::putText(img, "Iteration number", cv::Point(draw_size / 2, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
        char max_batches_buff[100];
        sprintf(max_batches_buff, "in cfg max_batches=%d", max_batches);
        cv::putText(img, max_batches_buff, cv::Point(draw_size - 195, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
        cv::putText(img, "Press 's' to save : chart.png", cv::Point(5, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
        if (!dont_show) {
            printf(" If error occurs - run training with flag: -dont_show \n");
            cv::namedWindow("average loss", cv::WINDOW_NORMAL);
            cv::moveWindow("average loss", 0, 0);
            cv::resizeWindow("average loss", img_size, img_size);
            cv::imshow("average loss", img);
            cv::waitKey(20);
        }
    }
    catch (...) {
        cerr << "OpenCV exception: draw_train_chart() \n";
    }
    return (mat_cv*)img_ptr;
}
// ----------------------------------------

void draw_train_loss(mat_cv* img_src, int img_size, float avg_loss, float max_img_loss, int current_batch, int max_batches,
    float precision, int draw_precision, char *accuracy_name, int dont_show, int mjpeg_port)
{
    try {
        cv::Mat &img = *(cv::Mat*)img_src;
        int img_offset = 60;
        int draw_size = img_size - img_offset;
        char char_buff[100];
        cv::Point pt1, pt2;
        pt1.x = img_offset + draw_size * (float)current_batch / max_batches;
        pt1.y = draw_size * (1 - avg_loss / max_img_loss);
        if (pt1.y < 0) pt1.y = 1;
        cv::circle(img, pt1, 1, CV_RGB(0, 0, 255), CV_FILLED, 8, 0);

        // precision
        if (draw_precision) {
            static float old_precision = 0;
            static float max_precision = 0;
            static int iteration_old = 0;
            static int text_iteration_old = 0;
            if (iteration_old == 0)
                cv::putText(img, accuracy_name, cv::Point(10, 12), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 0, 0), 1, CV_AA);

            cv::line(img,
                cv::Point(img_offset + draw_size * (float)iteration_old / max_batches, draw_size * (1 - old_precision)),
                cv::Point(img_offset + draw_size * (float)current_batch / max_batches, draw_size * (1 - precision)),
                CV_RGB(255, 0, 0), 1, 8, 0);

            sprintf(char_buff, "%2.1f%% ", precision * 100);
            cv::putText(img, char_buff, cv::Point(10, 28), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 5, CV_AA);
            cv::putText(img, char_buff, cv::Point(10, 28), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(200, 0, 0), 1, CV_AA);

            if ((std::fabs(old_precision - precision) > 0.1)  || (max_precision < precision) || (current_batch - text_iteration_old) >= max_batches / 10) {
                text_iteration_old = current_batch;
                max_precision = std::max(max_precision, precision);
                sprintf(char_buff, "%2.0f%% ", precision * 100);
                cv::putText(img, char_buff, cv::Point(pt1.x - 30, draw_size * (1 - precision) + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 5, CV_AA);
                cv::putText(img, char_buff, cv::Point(pt1.x - 30, draw_size * (1 - precision) + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(200, 0, 0), 1, CV_AA);
            }
            old_precision = precision;
            iteration_old = current_batch;
        }

        sprintf(char_buff, "current avg loss = %2.4f    iteration = %d", avg_loss, current_batch);
        pt1.x = 15, pt1.y = draw_size + 18;
        pt2.x = pt1.x + 460, pt2.y = pt1.y + 20;
        cv::rectangle(img, pt1, pt2, CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
        pt1.y += 15;
        cv::putText(img, char_buff, pt1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 100), 1, CV_AA);

        int k = 0;
        if (!dont_show) {
            cv::imshow("average loss", img);
            k = cv::waitKey(20);
        }
        static int old_batch = 0;
        if (k == 's' || current_batch == (max_batches - 1) || (current_batch / 100 > old_batch / 100)) {
            old_batch = current_batch;
            save_mat_png(img, "chart.png");
            cv::putText(img, "- Saved", cv::Point(260, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 0, 0), 1, CV_AA);
        }
        else
            cv::putText(img, "- Saved", cv::Point(260, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 1, CV_AA);

        if (mjpeg_port > 0) send_mjpeg((mat_cv *)&img, mjpeg_port, 500000, 100);
    }
    catch (...) {
        cerr << "OpenCV exception: draw_train_loss() \n";
    }
}
// ----------------------------------------


// ====================================================================
// Data augmentation
// ====================================================================
static box float_to_box_stride(float *f, int stride)
{
    box b = { 0 };
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}

image image_data_augmentation(mat_cv* mat, int w, int h,
    int pleft, int ptop, int swidth, int sheight, int flip,
    float dhue, float dsat, float dexp,
    int blur, int num_boxes, float *truth)
{
    image out;
    try {
        cv::Mat img = *(cv::Mat *)mat;

        // crop
        cv::Rect src_rect(pleft, ptop, swidth, sheight);
        cv::Rect img_rect(cv::Point2i(0, 0), img.size());
        cv::Rect new_src_rect = src_rect & img_rect;

        cv::Rect dst_rect(cv::Point2i(std::max<int>(0, -pleft), std::max<int>(0, -ptop)), new_src_rect.size());
        cv::Mat sized;

        if (src_rect.x == 0 && src_rect.y == 0 && src_rect.size() == img.size()) {
            cv::resize(img, sized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        }
        else {
            cv::Mat cropped(src_rect.size(), img.type());
            //cropped.setTo(cv::Scalar::all(0));
            cropped.setTo(cv::mean(img));

            img(new_src_rect).copyTo(cropped(dst_rect));

            // resize
            cv::resize(cropped, sized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        }

        // flip
        if (flip) {
            cv::Mat cropped;
            cv::flip(sized, cropped, 1);    // 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
            sized = cropped.clone();
        }

        // HSV augmentation
        // cv::COLOR_BGR2HSV, cv::COLOR_RGB2HSV, cv::COLOR_HSV2BGR, cv::COLOR_HSV2RGB
        if (dsat != 1 || dexp != 1 || dhue != 0) {
            if (img.channels() >= 3)
            {
                cv::Mat hsv_src;
                cvtColor(sized, hsv_src, cv::COLOR_RGB2HSV);    // RGB to HSV

                std::vector<cv::Mat> hsv;
                cv::split(hsv_src, hsv);

                hsv[1] *= dsat;
                hsv[2] *= dexp;
                hsv[0] += 179 * dhue;

                cv::merge(hsv, hsv_src);

                cvtColor(hsv_src, sized, cv::COLOR_HSV2RGB);    // HSV to RGB (the same as previous)
            }
            else
            {
                sized *= dexp;
            }
        }

        //std::stringstream window_name;
        //window_name << "augmentation - " << ipl;
        //cv::imshow(window_name.str(), sized);
        //cv::waitKey(0);

        if (blur) {
            cv::Mat dst(sized.size(), sized.type());
            if(blur == 1) cv::GaussianBlur(sized, dst, cv::Size(31, 31), 0);
            else {
                cv::Size kernel_size = cv::Size((blur / 2) * 2 + 1, (blur / 2) * 2 + 1);
                cv::GaussianBlur(sized, dst, kernel_size, 0);

                // sharpen
                //cv::Mat img_tmp;
                //cv::GaussianBlur(dst, img_tmp, cv::Size(), 3);
                //cv::addWeighted(dst, 1.5, img_tmp, -0.5, 0, img_tmp);
                //dst = img_tmp;
            }
            //std::cout << " blur num_boxes = " << num_boxes << std::endl;

            if (blur == 1) {
                cv::Rect img_rect(0, 0, sized.cols, sized.rows);
                int t;
                for (t = 0; t < num_boxes; ++t) {
                    box b = float_to_box_stride(truth + t*(4 + 1), 1);
                    if (!b.x) break;
                    int left = (b.x - b.w / 2.)*sized.cols;
                    int width = b.w*sized.cols;
                    int top = (b.y - b.h / 2.)*sized.rows;
                    int height = b.h*sized.rows;
                    cv::Rect roi(left, top, width, height);
                    roi = roi & img_rect;

                    sized(roi).copyTo(dst(roi));
                }
            }
            dst.copyTo(sized);
        }

        // Mat -> image
        out = mat_to_image(sized);
    }
    catch (...) {
        cerr << "OpenCV can't augment image: " << w << " x " << h << " \n";
        out = mat_to_image(*(cv::Mat*)mat);
    }
    return out;
}

// blend two images with (alpha and beta)
void blend_images_cv(image new_img, float alpha, image old_img, float beta)
{
    cv::Mat new_mat(cv::Size(new_img.w, new_img.h), CV_32FC(new_img.c), new_img.data);// , size_t step = AUTO_STEP)
    cv::Mat old_mat(cv::Size(old_img.w, old_img.h), CV_32FC(old_img.c), old_img.data);
    cv::addWeighted(new_mat, alpha, old_mat, beta, 0.0, new_mat);
}

// ====================================================================
// Show Anchors
// ====================================================================
void show_acnhors(int number_of_boxes, int num_of_clusters, float *rel_width_height_array, model anchors_data, int width, int height)
{
    cv::Mat labels = cv::Mat(number_of_boxes, 1, CV_32SC1);
    cv::Mat points = cv::Mat(number_of_boxes, 2, CV_32FC1);
    cv::Mat centers = cv::Mat(num_of_clusters, 2, CV_32FC1);

    for (int i = 0; i < number_of_boxes; ++i) {
        points.at<float>(i, 0) = rel_width_height_array[i * 2];
        points.at<float>(i, 1) = rel_width_height_array[i * 2 + 1];
    }

    for (int i = 0; i < num_of_clusters; ++i) {
        centers.at<float>(i, 0) = anchors_data.centers.vals[i][0];
        centers.at<float>(i, 1) = anchors_data.centers.vals[i][1];
    }

    for (int i = 0; i < number_of_boxes; ++i) {
        labels.at<int>(i, 0) = anchors_data.assignments[i];
    }

    size_t img_size = 700;
    cv::Mat img = cv::Mat(img_size, img_size, CV_8UC3);

    for (int i = 0; i < number_of_boxes; ++i) {
        cv::Point pt;
        pt.x = points.at<float>(i, 0) * img_size / width;
        pt.y = points.at<float>(i, 1) * img_size / height;
        int cluster_idx = labels.at<int>(i, 0);
        int red_id = (cluster_idx * (uint64_t)123 + 55) % 255;
        int green_id = (cluster_idx * (uint64_t)321 + 33) % 255;
        int blue_id = (cluster_idx * (uint64_t)11 + 99) % 255;
        cv::circle(img, pt, 1, CV_RGB(red_id, green_id, blue_id), CV_FILLED, 8, 0);
        //if(pt.x > img_size || pt.y > img_size) printf("\n pt.x = %d, pt.y = %d \n", pt.x, pt.y);
    }

    for (int j = 0; j < num_of_clusters; ++j) {
        cv::Point pt1, pt2;
        pt1.x = pt1.y = 0;
        pt2.x = centers.at<float>(j, 0) * img_size / width;
        pt2.y = centers.at<float>(j, 1) * img_size / height;
        cv::rectangle(img, pt1, pt2, CV_RGB(255, 255, 255), 1, 8, 0);
    }
    save_mat_png(img, "cloud.png");
    cv::imshow("clusters", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

}   // extern "C"


#else  // OPENCV
int wait_key_cv(int delay) { return 0; }
int wait_until_press_key_cv() { return 0; }
void destroy_all_windows_cv() {}
#endif // OPENCV
