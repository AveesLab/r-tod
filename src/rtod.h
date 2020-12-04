#ifndef RTOD_H
#define RTOD_H

#include "image.h"

#if (defined ZERO_SLACK)
#define CYCLE_OFFSET 1000
#else
#define CYCLE_OFFSET 25
#endif

/* Measurement */
#define MEASUREMENT_PATH "measure"
#define MEASUREMENT_FILE "/measure.csv"
#define OBJ_DET_CYCLE_IDX 1000

#define QLEN 4
#define NFRAMES 3

#define MAX(x,y) (((x) < (y) ? (y) : (x)))
#define MIN(x,y) (((x) < (y) ? (x) : (y)))
/* calculate inter frame gap */
#define GET_IFG(x,y) ((x) - (y)); \
    (y) = (x);


#ifdef __cplusplus
extern "C" {
#endif
    void rtod(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
            int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host, int benchmark, int benchmark_layers, int w, int h, int cam_fps);

#ifdef __cplusplus
}
#endif

struct frame_data frame[3]; // v4l2 image data

double e_fetch_array[OBJ_DET_CYCLE_IDX];
double b_fetch_array[OBJ_DET_CYCLE_IDX];
double d_fetch_array[OBJ_DET_CYCLE_IDX];
double e_infer_cpu_array[OBJ_DET_CYCLE_IDX];
double e_infer_gpu_array[OBJ_DET_CYCLE_IDX];
double d_infer_array[OBJ_DET_CYCLE_IDX];
double e_disp_array[OBJ_DET_CYCLE_IDX];
double b_disp_array[OBJ_DET_CYCLE_IDX];
double d_disp_array[OBJ_DET_CYCLE_IDX];
double slack[OBJ_DET_CYCLE_IDX];
double fps_array[OBJ_DET_CYCLE_IDX];
double e2e_delay[OBJ_DET_CYCLE_IDX];
int inter_frame_gap_array[OBJ_DET_CYCLE_IDX];
double cycle_time_array[OBJ_DET_CYCLE_IDX];
int num_object_array[OBJ_DET_CYCLE_IDX];
double transfer_delay_array[OBJ_DET_CYCLE_IDX];

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
int measure;

int frame_sequence_tmp;
int inter_frame_gap;

double start_infer;
double end_infer;
double d_infer;
double d_disp;
double start_disp;
double end_disp;
double start_fetch;
double end_fetch;
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

void push_data(void);
int get_fetch_offset(void);
int write_result(void);
double get_time_in_ms(void);
int check_on_demand(void);
void *rtod_fetch_thread(void *ptr);
void *rtod_inference_thread(void *ptr);
void *rtod_display_thread(void *ptr);

#endif
