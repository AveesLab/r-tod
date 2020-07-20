#ifndef DEMO_H
#define DEMO_H

#include "image.h"

#if (defined ZERO_SLACK)
#define cycle_offset 500
#else
#define cycle_offset 25
#endif

#define obj_det_cycle_idx 1000
#define cycle 1
#define QLEN 4
#define NFRAMES 3

#define MAX(x,y) (((x) < (y) ? (y) : (x)))
#define MIN(x,y) (((x) < (y) ? (x) : (y)))
/* calculate inter frame gap */
#define GET_IFG(x,y) ((x) - (y)); \
    (y) = (x);

/* CPU & GPU information */
#define NUM_TRACE 4 /* GPU power, CPU power, GPU temperature, CPU temperature */
#define GPU_POWER_PATH "/sys/bus/i2c/devices/1-0040/iio_device/in_power0_input"
#define CPU_POWER_PATH "/sys/bus/i2c/devices/1-0040/iio_device/in_power1_input"
#define GPU_TEMP_PATH "/sys/devices/virtual/thermal/thermal_zone1/temp"
#define CPU_TEMP_PATH "/sys/devices/virtual/thermal/thermal_zone0/temp"

/* Measurement */
#define MEASUREMENT_PATH "measure"
#define MEASUREMENT_FILE "/measure.csv"

#ifdef __cplusplus
extern "C" {
#endif
void r_tod(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
    int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host, int benchmark, int benchmark_layers, int offset);

#ifdef __cplusplus
}
#endif

#ifdef TRACE
/* Check first obj_det_cycle_idx */
int first_trace;

/* Trace obj_det_cycle_idx */
int trace_iter;
#endif

struct frame_data frame[3]; // v4l2 image data

double e_fetch_array[obj_det_cycle_idx];
double b_fetch_array[obj_det_cycle_idx];
double d_fetch_array[obj_det_cycle_idx];
double e_infer_cpu_array[obj_det_cycle_idx];
double e_infer_gpu_array[obj_det_cycle_idx];
double d_infer_array[obj_det_cycle_idx];
double e_disp_array[obj_det_cycle_idx];
double b_disp_array[obj_det_cycle_idx];
double d_disp_array[obj_det_cycle_idx];
double slack[obj_det_cycle_idx];
double fps_array[obj_det_cycle_idx];
double e2e_delay[obj_det_cycle_idx];
int inter_frame_gap_array[obj_det_cycle_idx];
double cycle_time_array[obj_det_cycle_idx];
int num_object_array[obj_det_cycle_idx];
double transfer_delay_array[obj_det_cycle_idx];
double trace_array[NUM_TRACE][obj_det_cycle_idx];

double e_fetch_sum;
double b_fetch_sum;
double d_fetch_sum;
double e_infer_cpu_sum;
double e_infer_gpu_sum;
double d_infer_sum;
double e_disp_sum;
double b_disp_sum;
double d_disp_sum;
double slack_sum;
double e2e_delay_sum;
double fps_sum;
double cycle_time_sum;
double inter_frame_gap_sum;
double num_object_sum;
double trace_data_sum;

#ifdef ZERO_SLACK
double s_min;
double e_fetch_max;
double b_fetch_max;
#endif

double frame_timestamp[3];
int buff_index;
int sleep_time;
int cnt;
int display_index;
int detect_index;
int fetch_offset; // zero slack
int cycle_index;
double cycle_array[QLEN];
int ondemand;
int num_object;

int frame_sequence_tmp;
int inter_frame_gap;

double infer_start;
double infer_end;
double d_infer;
double d_disp;
double display_end;
double fetch_start;
double d_fetch;
double e_fetch;
double b_fetch;
double image_waiting_time;
double select_time;
double slack_time;
double cycle_end;
double transfer_delay;
double draw_bbox_time;
double waitkey_start;
double e_infer_gpu;
double b_disp;

char **demo_names;
image **demo_alphabet;
int demo_classes;

int nboxes;
detection *dets;

network net;
image in_s ;
image det_s;

cap_cv *cap;
float fps;
float demo_thresh;
int demo_ext_output;
long long int frame_id;
int demo_json_port;


float* predictions[NFRAMES];
int demo_index;
mat_cv* cv_images[NFRAMES];
float *avg;

mat_cv* in_img;
mat_cv* det_img;
mat_cv* show_img;

volatile int flag_exit;
int letter_box;

int get_rand_offset(int a, int b);
double gettime_after_boot();
int check_on_demand();
void *fetch_in_thread(void *ptr);
void *detect_in_thread(void *ptr);
double get_wall_time();

#endif
