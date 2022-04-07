#define _GNU_SOURCE
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "rtod.h"
#include "darknet.h"

#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif


#ifdef V4L2
#include "v4l2.h"
#endif

#ifdef OPENCV

#include "http_stream.h"

//=========================================================================//
extern int buff_index=0;
extern int cnt = 0;
extern double cycle_array[QLEN] = {0,};
extern int ondemand = 1;
extern int num_object = 0;
extern int measure = 1;

extern int nboxes = 0;
extern detection *dets = NULL;

extern float fps = 0;
extern float demo_thresh = 0;
extern int demo_ext_output = 0;
extern long long int frame_id = 0;
extern int demo_json_port = -1;
extern int demo_index = 0;
extern int letter_box = 0;
extern int fetch_offset = 0; // zero slack

extern double e_fetch_sum = 0;
extern double b_fetch_sum = 0;
extern double d_fetch_sum = 0;
extern double e_infer_cpu_sum = 0;
extern double e_infer_gpu_sum = 0;
extern double d_infer_sum = 0;
extern double e_disp_sum = 0;
extern double b_disp_sum = 0;
extern double d_disp_sum = 0;
extern double slack_sum = 0;
extern double e2e_delay_sum = 0;
extern double fps_sum = 0;
extern double cycle_time_sum = 0;
extern double inter_frame_gap_sum = 0;
extern double num_object_sum = 0;
extern double trace_data_sum = 0;

int *fd_handler = NULL;

#ifndef ZERO_SLACK
int contention_free = 1;
#endif

#ifdef ZERO_SLACK
extern double s_min = 10000.;
extern double e_fetch_max = 0;
extern double b_fetch_max = 0;

int contention_free = 0;
#endif


/* Save result in csv*/
int write_result(void)
{
    static int exist=0;
    FILE *fp;
    char file_path[256] = "";
    int tick = 0;

    strcat(file_path, MEASUREMENT_PATH);
    strcat(file_path, MEASUREMENT_FILE);

    fp=fopen(file_path,"w+");

    if (fp == NULL) 
    {
        /* make directory */
        while(!exist)
        {
            int result;

            usleep(10 * 1000);

            result = mkdir(MEASUREMENT_PATH, 0766);
            if(result == 0) { 
                exist = 1;

                fp=fopen(file_path,"w+");
            }

            if(tick == 100)
            {
                fprintf(stderr, "\nERROR: Fail to Create %s\n", file_path);

                return -1;
            }
            else tick++;
        }
    }
    else printf("\nWrite output in %s\n", file_path); 

    fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", "e_fetch", "b_fetch", "d_fetch",
            "e_infer", "b_infer", "d_infer", "e_disp", "b_disp", "d_disp",
            "slack", "e2e_delay", "fps", "c_sys", "IFG", "n_obj");

    for(int i=0;i<OBJ_DET_CYCLE_IDX;i++)
    {
        e_fetch_sum += e_fetch_array[i];
        b_fetch_sum += b_fetch_array[i];
        d_fetch_sum += d_fetch_array[i];
        e_infer_cpu_sum += e_infer_cpu_array[i];
        e_infer_gpu_sum += e_infer_gpu_array[i];
        d_infer_sum += d_infer_array[i];
        e_disp_sum += e_disp_array[i];
        b_disp_sum += b_disp_array[i];
        d_disp_sum += d_disp_array[i];
        slack_sum += slack[i];
        e2e_delay_sum += e2e_delay[i];
        fps_sum += fps_array[i];
        cycle_time_sum += cycle_time_array[i];
        inter_frame_gap_sum += (double)inter_frame_gap_array[i];
        num_object_sum += (double)num_object_array[i];

        fprintf(fp, "%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%d,%d\n", e_fetch_array[i], b_fetch_array[i],d_fetch_array[i], 
                e_infer_cpu_array[i], e_infer_gpu_array[i], d_infer_array[i], e_disp_array[i], b_disp_array[i], d_disp_array[i], 
                slack[i], e2e_delay[i], fps_array[i], cycle_time_array[i], inter_frame_gap_array[i], num_object_array[i]);
    }
    fclose(fp);

    return 1;
}

#ifdef ZERO_SLACK
/* Calculate fetch offset */
int get_fetch_offset(void)
{
    int offset;
    int tick = 0;

    if (cnt < (CYCLE_OFFSET - 1))
    {
        printf("\nMIN cycle time : %f\n", MIN(s_min, (1000./fps)));
        printf("MAX e_fetch : %f\n", MAX(e_fetch_max, e_fetch));
        printf("MAX b_fetch : %f\n", MAX(b_fetch_max, b_fetch));

        s_min = MIN(s_min, (1000./fps));
        e_fetch_max = MAX(e_fetch_max, e_fetch);
        b_fetch_max = MAX(b_fetch_max, b_fetch);
    }
    else if (cnt == (CYCLE_OFFSET - 1)) 
    {
        offset = (int)(s_min - e_fetch_max - b_fetch_max);

        if (offset < 0) offset = 0;

        printf("Calculated fetch offset: %d ms\n"
                " Enter the fetch offset (ms): ", offset);

        if (-1 == scanf("%d", &fetch_offset))
        {
            perror("Invalid fetch offset");
            return -1;

        }
        else 
        {
            cnt = 0;
            measure = 0;
        }
    }
    else return 0; 

    return 1;
}
#endif

void push_data(void)
{
    b_fetch_array[cnt - CYCLE_OFFSET] = b_fetch;
    e_fetch_array[cnt - CYCLE_OFFSET] = d_fetch - b_fetch - fetch_offset;
    //e_fetch_array[cnt - CYCLE_OFFSET] = d_fetch - b_fetch - fetch_offset;
    d_fetch_array[cnt - CYCLE_OFFSET] = d_fetch;
    inter_frame_gap_array[cnt - CYCLE_OFFSET] = inter_frame_gap;
    transfer_delay_array[cnt - CYCLE_OFFSET] = transfer_delay;

    e_infer_cpu_array[cnt - CYCLE_OFFSET] = d_infer - e_infer_gpu;
    e_infer_gpu_array[cnt - CYCLE_OFFSET] = e_infer_gpu;
    d_infer_array[cnt - CYCLE_OFFSET] = d_infer;

    fps_array[cnt - CYCLE_OFFSET] = fps;
    cycle_time_array[cnt - CYCLE_OFFSET] = 1000./fps;
    e2e_delay[cnt - CYCLE_OFFSET] = end_disp - frame[display_index].frame_timestamp;
    e_disp_array[cnt - CYCLE_OFFSET] = d_disp - b_disp;
    b_disp_array[cnt - CYCLE_OFFSET] = b_disp;
    d_disp_array[cnt - CYCLE_OFFSET] = d_disp;
    slack[cnt - CYCLE_OFFSET] = slack_time;
    num_object_array[cnt - CYCLE_OFFSET] = num_object;

    //printf("num_object : %d\n", num_object);
    //printf("slack: %f\n",slack[cnt-CYCLE_OFFSET]);
    printf("latency: %f\n",e2e_delay[cnt - CYCLE_OFFSET]);
    printf("cnt : %d\n",cnt);

    return;
}

/* Timestamp in ms */
double get_time_in_ms(void)
{
    struct timespec time_after_boot;
    clock_gettime(CLOCK_MONOTONIC,&time_after_boot);
    return (time_after_boot.tv_sec*1000+time_after_boot.tv_nsec*0.000001);
}

/* Check if On-demand capture */
int check_on_demand(void)
{
    int env_var_int;
    char *env_var;
    static int size;
    int on_demand;

#if (defined V4L2)
    env_var = getenv("V4L2_QLEN");
#else
    env_var = getenv("OPENCV_QLEN");
#endif

    if(env_var != NULL){
        env_var_int = atoi(env_var);
    }
    else {
        printf("Using DEFAULT V4L Queue Length\n");
        env_var_int = 4;
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

    return on_demand;
}
//=====================================================================//


/*
static int nboxes = 0;
static detection *dets = NULL;

static network net;
static image in_s ;
static image det_s;

static cap_cv *cap;
static float fps = 0;
static float demo_thresh = 0;
static int demo_ext_output = 0;
static long long int frame_id = 0;
static int demo_json_port = -1;

#define NFRAMES 3

static float* predictions[NFRAMES];
static int demo_index = 0;
static mat_cv* cv_images[NFRAMES];
static float *avg;

mat_cv* in_img;
mat_cv* det_img;
mat_cv* show_img;

static volatile int flag_exit;
static int letter_box = 0;
*/
void *fetch_in_thread(void *ptr)
{
    start_fetch = get_time_in_ms();

    usleep(fetch_offset * 1000);

    int dont_close_stream = 0;    // set 1 if your IP-camera periodically turns off and turns on video-stream
    if(letter_box)
        in_s = get_image_from_stream_letterbox(cap, net.w, net.h, net.c, &in_img, dont_close_stream);
    else
        in_s = in_s = get_image_from_stream_resize_with_timestamp(cap, net.w, net.h, net.c, &in_img, dont_close_stream, &frame[buff_index]);
    if(!in_s.data){
        printf("Stream closed.\n");
        flag_exit = 1;
        //exit(EXIT_FAILURE);
        return 0;
    }
    //in_s = resize_image(in, net.w, net.h);

    end_fetch = get_time_in_ms();

    image_waiting_time = frame[buff_index].frame_timestamp - start_fetch;
    image_waiting_time -= fetch_offset;

    if(ondemand) transfer_delay = frame[buff_index].select - image_waiting_time;
    else transfer_delay = .0; 

    inter_frame_gap = GET_IFG(frame[buff_index].frame_sequence, frame_sequence_tmp);

    if(cnt >= (CYCLE_OFFSET - 5)){
        d_fetch = end_fetch - start_fetch;
        b_fetch = frame[buff_index].select;
        e_fetch = d_fetch - b_fetch - fetch_offset;
    }
    return 0;
}

void *detect_in_thread(void *ptr)
{
    
    start_infer = get_time_in_ms();
    layer l = net.layers[net.n-1];
    float *X = det_s.data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, NFRAMES, l.outputs, avg);
    l.output = avg;

    cv_images[demo_index] = det_img;
    det_img = cv_images[(demo_index + NFRAMES / 2 + 1) % NFRAMES];
    demo_index = (demo_index + 1) % NFRAMES;

    if (letter_box)
        dets = get_network_boxes(&net, get_width_mat(in_img), get_height_mat(in_img), demo_thresh, demo_thresh, 0, 1, &nboxes, 1); // letter box
    else
        dets = get_network_boxes(&net, net.w, net.h, demo_thresh, demo_thresh, 0, 1, &nboxes, 0); // resized

    end_infer = get_time_in_ms();

    d_infer = end_infer - start_infer;

    return 0;
}

double get_wall_time()
{
    struct timeval walltime;
    if (gettimeofday(&walltime, NULL)) {
        return 0;
    }
    return (double)walltime.tv_sec + (double)walltime.tv_usec * .000001;
}

void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
    int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host,
    int benchmark, int benchmark_layers)
{
    letter_box = letter_box_in;
    in_img = det_img = show_img = NULL;
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_ext_output = ext_output;
    demo_json_port = json_port;
    printf("Demo\n");
    net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if(weightfile){
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = get_capture_video_stream(filename);
    }else{
        printf("Webcam index: %d\n", cam_index);
        cap = get_capture_webcam(cam_index);
    }

    if (!cap) {
#ifdef WIN32
        printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
#endif
        error("Couldn't connect to webcam.\n");
    }

    layer l = net.layers[net.n-1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < NFRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

    if (l.classes != demo_classes) {
        printf("\n Parameters don't match: in cfg-file classes=%d, in data-file classes=%d \n", l.classes, demo_classes);
        getchar();
        exit(0);
    }


    flag_exit = 0;

    pthread_t fetch_thread;
    pthread_t detect_thread;
    
    printf("ondemand : %d\n", ondemand);

    fetch_in_thread(0);
    det_img = in_img;
    det_s = in_s;

    fetch_in_thread(0);
    detect_in_thread(0);
    det_img = in_img;
    det_s = in_s;

    for (j = 0; j < NFRAMES / 2; ++j) {
        free_detections(dets, nboxes);
        fetch_in_thread(0);
        detect_in_thread(0);
        det_img = in_img;
        det_s = in_s;
    }

    int count = 0;
    if(!prefix && !dont_show){
        int full_screen = 0;
        create_window_cv("Demo", full_screen, 1352, 1013);
    }


    write_cv* output_video_writer = NULL;
    if (out_filename && !flag_exit)
    {
        int src_fps = 25;
        src_fps = get_stream_fps_cpp_cv(cap);
        output_video_writer =
            create_video_writer(out_filename, 'D', 'I', 'V', 'X', src_fps, get_width_mat(det_img), get_height_mat(det_img), 1);

        //'H', '2', '6', '4'
        //'D', 'I', 'V', 'X'
        //'M', 'J', 'P', 'G'
        //'M', 'P', '4', 'V'
        //'M', 'P', '4', '2'
        //'X', 'V', 'I', 'D'
        //'W', 'M', 'V', '2'
    }

    int send_http_post_once = 0;
    const double start_time_lim = get_time_point();
    double before = get_time_point();
    double start_time = get_time_point();
    float avg_fps = 0;
    int frame_counter = 0;

    while(1){
        ++count;
        {

            display_index = (buff_index + 1) %3;
            detect_index = (buff_index + 2) %3;

            const float nms = .45;    // 0.4F
            int local_nboxes = nboxes;
            detection *local_dets = dets;

            if (!benchmark) if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

            double start_disp = get_time_in_ms();

            //if (nms) do_nms_obj(local_dets, local_nboxes, l.classes, nms);    // bad results
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(local_dets, local_nboxes, l.classes, nms);
                else diounms_sort(local_dets, local_nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }

            //printf("\033[2J");
            //printf("\033[1;1H");
            //printf("\nFPS:%.1f\n", fps);
            printf("Objects:\n\n");

            ++frame_id;
            if (demo_json_port > 0) {
                int timeout = 400000;
                send_json(local_dets, local_nboxes, l.classes, demo_names, frame_id, demo_json_port, timeout);
            }

            //char *http_post_server = "webhook.site/898bbd9b-0ddd-49cf-b81d-1f56be98d870";
            if (http_post_host && !send_http_post_once) {
                int timeout = 3;            // 3 seconds
                int http_post_port = 80;    // 443 https, 80 http
                if (send_http_post_request(http_post_host, http_post_port, filename,
                    local_dets, nboxes, classes, names, frame_id, ext_output, timeout))
                {
                    if (time_limit_sec > 0) send_http_post_once = 1;
                }
            }

            if (!benchmark) draw_detections_cv_v3(show_img, local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);

            draw_bbox_time = get_time_in_ms() - start_disp;

            free_detections(local_dets, local_nboxes);

            printf("\nFPS:%.1f \t AVG_FPS:%.1f\n", fps, avg_fps);

            if(!prefix){
                if (!dont_show) {
                    show_image_mat(show_img, "Demo");

                    //waitkey_start = get_time_in_ms();
		    //int c=0;
                    //cudaDeviceSynchronize();
                    waitkey_start = get_time_in_ms();
		    int c = wait_key_cv(1);
                    b_disp = get_time_in_ms() - waitkey_start;

                    if (c == 10) {
                        if (frame_skip == 0) frame_skip = 60;
                        else if (frame_skip == 4) frame_skip = 0;
                        else if (frame_skip == 60) frame_skip = 4;
                        else frame_skip = 0;
                    }
                    else if (c == 27 || c == 1048603) // ESC - exit (OpenCV 2.x / 3.x)
                    {
                        flag_exit = 1;
                    }
                }
            }else{
                char buff[256];
                sprintf(buff, "%s_%08d.jpg", prefix, count);
                if(show_img) save_cv_jpg(show_img, buff);
            }

            // if you run it with param -mjpeg_port 8090  then open URL in your web-browser: http://localhost:8090
            if (mjpeg_port > 0 && show_img) {
                int port = mjpeg_port;
                int timeout = 400000;
                int jpeg_quality = 40;    // 1 - 100
                send_mjpeg(show_img, port, timeout, jpeg_quality);
            }

            // save video file
            if (output_video_writer && show_img) {
                write_frame_cv(output_video_writer, show_img);
                printf("\n cvWriteFrame \n");
            }

            /* display end */

            end_disp = get_time_in_ms();

            d_disp = end_disp - start_disp; 

            pthread_join(detect_thread, 0);

            if (!benchmark) {
                pthread_join(fetch_thread, 0);
                free_image(det_s);
            }

            if (time_limit_sec > 0 && (get_time_point() - start_time_lim)/1000000 > time_limit_sec) {
                printf(" start_time_lim = %f, get_time_point() = %f, time spent = %f \n", start_time_lim, get_time_point(), get_time_point() - start_time_lim);
                break;
            }

            if (flag_exit == 1) break;

            if(delay == 0){
                if(!benchmark) release_mat(&show_img);
                show_img = det_img;
            }
            det_img = in_img;
            det_s = in_s;
        }
        --delay;
        if(delay < 0){
            delay = frame_skip;

            //double after = get_wall_time();
            //float curr = 1./(after - before);
            double after = get_time_point();    // more accurate time measurements
            float curr = 1000000. / (after - before);
            fps = fps*0.9 + curr*0.1;
            before = after;

            float spent_time = (get_time_point() - start_time) / 1000000;
            frame_counter++;
            if (spent_time >= 3.0f) {
                //printf(" spent_time = %f \n", spent_time);
                avg_fps = frame_counter / spent_time;
                frame_counter = 0;
                start_time = get_time_point();
            }
        }
        cycle_array[cycle_index] = 1000./fps;
        cycle_index = (cycle_index + 1) % 4;


#ifdef MEASUREMENT
        if (cnt >= CYCLE_OFFSET) push_data();

        /* Exit object detection cycle */
        if(cnt == ((OBJ_DET_CYCLE_IDX + CYCLE_OFFSET) - 1)) 
        {
            if(-1 == write_result())
            {
                /* return error */
                exit(0);
            }

            /* exit loop */
            break;
        }
#endif

        /* Increase count */
        if(cnt != ((OBJ_DET_CYCLE_IDX + CYCLE_OFFSET)-1)) cnt++;
        /* Change buffer index */
        buff_index = (buff_index + 1) % 3;
    }
    cnt = 0;

#ifdef MEASUREMENT
    /* Average data */
    printf("============ Darknet(2020) data ============\n");
    printf("Avg fetch execution time (ms) : %0.2f\n", e_fetch_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg fetch blocking time (ms) : %0.2f\n", b_fetch_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg fetch delay (ms) : %0.2f\n", d_fetch_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg infer execution on cpu (ms) : %0.2f\n", e_infer_cpu_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg infer execution on gpu (ms) : %0.2f\n", e_infer_gpu_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg infer delay (ms) : %0.2f\n", d_infer_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg disp execution time (ms) : %0.2f\n", e_disp_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg disp blocking time (ms) : %0.2f\n", b_disp_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg disp delay (ms) : %0.2f\n", d_disp_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg salck (ms) : %0.2f\n", slack_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg E2E delay (ms) : %0.2f\n", e2e_delay_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg cycle time (ms) : %0.2f\n", cycle_time_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg inter frame gap : %0.2f\n", inter_frame_gap_sum / OBJ_DET_CYCLE_IDX);
    printf("Avg number of object : %0.2f\n", num_object_sum / OBJ_DET_CYCLE_IDX);
    printf("=====================================\n");
#endif
    printf("input video stream closed. \n");
    if (output_video_writer) {
        release_video_writer(&output_video_writer);
        printf("output_video_writer closed. \n");
    }

    // free memory
    free_image(in_s);
    free_detections(dets, nboxes);

    free(avg);
    for (j = 0; j < NFRAMES; ++j) free(predictions[j]);
    demo_index = (NFRAMES + demo_index - 1) % NFRAMES;
    for (j = 0; j < NFRAMES; ++j) {
            release_mat(&cv_images[j]);
    }

    free_ptrs((void **)names, net.layers[net.n - 1].classes);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);
    free_network(net);
    //cudaProfilerStop();
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
    int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host,
    int benchmark, int benchmark_layers)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif
