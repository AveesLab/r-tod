#define _GNU_SOURCE
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "image_opencv.h"
#include "r_tod.h"
#include "darknet.h"
#include "v4l2.h"
#include <sched.h>

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

#ifdef ZERO_SLACK
extern double s_min = 10000.;
extern double e_fetch_max = 0;
extern double b_fetch_max = 0;
#endif

extern mat_cv* in_img;
extern mat_cv* det_img;
extern mat_cv* show_img;
extern image in_s ;
extern image det_s;

double gettime_after_boot()
{
    struct timespec time_after_boot;
    clock_gettime(CLOCK_MONOTONIC,&time_after_boot);
    return (time_after_boot.tv_sec*1000+time_after_boot.tv_nsec*0.000001);
}

int check_on_demand()
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

void r_tod(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
        int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host,
        int benchmark, int benchmark_layers, int offset)
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

    fetch_offset = offset;

    if(filename){
        printf("video file: %s\n", filename);
        cap = get_capture_video_stream(filename);
    }else{
        printf("Webcam index: %d\n", cam_index);
#ifdef V4L2
        char cam_dev[20] = "/dev/video";
        char index[2];
        sprintf(index, "%d", cam_index);
        strcat(cam_dev, index);
        printf("cam dev : %s\n", cam_dev);

        int frames = 30;
        int w = 640;
        int h = 480;
        if(open_device(cam_dev, frames, w, h) < 0)
        {
            error("Couldn't connect to webcam.\n");

        }
#else
        cap = get_capture_webcam(cam_index);

        if (!cap) {
#ifdef WIN32
            printf("Check that you have copied file opencv_ffmpeg340_64.dll to the same directory where is darknet.exe \n");
#endif
            error("Couldn't connect to webcam.\n");
        }
#endif
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
    pthread_t trace_thread;

    ondemand = check_on_demand();

    if(ondemand != 1) { 
        fprintf(stderr, "ERROR : R-TOD needs on-demand capture.\n");
        exit(0);
    }
    //printf("ondemand : %d\n", ondemand);

#ifdef V4L2
    frame[0].frame = capture_image(&frame[buff_index]);
    frame[0].resize_frame = letterbox_image(frame[0].frame, net.w, net.h);

    frame[1].frame = frame[0].frame;
    frame[1].resize_frame = letterbox_image(frame[0].frame, net.w, net.h);

    frame[2].frame = frame[0].frame;
    frame[2].resize_frame = letterbox_image(frame[0].frame, net.w, net.h);

#else
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
#endif

    int count = 0;
    if(!prefix && !dont_show){
        int full_screen = 0;
        create_window_cv("Demo", full_screen, 1352, 1013);
        //make_window("Demo", 1352, 1013, full_screen);
    }

    write_cv* output_video_writer = NULL;
    if (out_filename && !flag_exit)
    {
        int src_fps = 25;
        src_fps = get_stream_fps_cpp_cv(cap);
#ifndef V4L2
        output_video_writer =
            create_video_writer(out_filename, 'D', 'I', 'V', 'X', src_fps, get_width_mat(det_img), get_height_mat(det_img), 1);
#endif

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
    double before_1 = gettime_after_boot();
    double start_time = get_time_point();
    float avg_fps = 0;
    int frame_counter = 0;

    while(1){
        printf("================start================\n");
        ++count;
        {
            /* Image index */
            display_index = (buff_index + 2) %3;
            detect_index = (buff_index) %3;

            const float nms = .45;    // 0.4F
            int local_nboxes = nboxes;
            detection *local_dets = dets;

            /* Fork fetch thread */
            if (!benchmark) if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");

            /* display thread */

            double display_start = gettime_after_boot();

            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(local_dets, local_nboxes, l.classes, nms);
                else diounms_sort(local_dets, local_nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }

#ifdef V4L2
            //if (!benchmark) draw_detections_v3(display, local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);

            if (!benchmark) draw_detections_v3(frame[display_index].frame, local_dets, local_nboxes, demo_thresh, demo_names, demo_alphabet, demo_classes, demo_ext_output);
            free_detections(local_dets, local_nboxes);

            draw_bbox_time = gettime_after_boot() - display_start;

            /* Image display */
            display_in_thread(0);
#else
            /* original display thread */

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

            draw_bbox_time = gettime_after_boot() - display_start;

            free_detections(local_dets, local_nboxes);

            printf("\nFPS:%.1f \t AVG_FPS:%.1f\n", fps, avg_fps);

            if(!prefix){
                if (!dont_show) {
                    show_image_mat(show_img, "Demo");

                    waitkey_start = gettime_after_boot();
                    int c = wait_key_cv(1);
                    b_disp = gettime_after_boot() - waitkey_start;

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
#endif
            /* display end */

            display_end = gettime_after_boot();

            d_disp = display_end - display_start; 

            /* Join fetch thread */
            if (!benchmark) {
                pthread_join(fetch_thread, 0);
                free_image(det_s);
            }

            /* Fork inference thread */
            det_img = in_img;
            det_s = in_s;

            detect_in_thread(0);

            if (time_limit_sec > 0 && (get_time_point() - start_time_lim)/1000000 > time_limit_sec) {
                printf(" start_time_lim = %f, get_time_point() = %f, time spent = %f \n", start_time_lim, get_time_point(), get_time_point() - start_time_lim);
                break;
            }

            if (flag_exit == 1) break;

            if(delay == 0){
#ifndef V4L2
                if(!benchmark) release_mat(&show_img);
#endif
                show_img = det_img;
            }
            cycle_end = gettime_after_boot();
        }
        --delay;
        if(delay < 0){
            delay = frame_skip;
            //double after = get_wall_time();
            //float curr = 1./(after - before);
            double after = get_time_point();    // more accurate time measurements
            double after_1 = gettime_after_boot();    
            float curr = 1000000. / (after - before);
            float curr_1 = (after_1 - before_1);
            printf("FPS : %f\n",1000.0/curr_1);
            //fps = fps*0.9 + curr*0.1;  
            fps = 1000.0/curr_1;
            before = after;
            before_1 = after_1;

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
        slack_time = (MAX(d_infer, d_disp)) - (fetch_offset + d_fetch);

        if(cnt >= cycle_offset){
            fps_array[cnt-cycle_offset] = fps;
            cycle_time_array[cnt-cycle_offset] = 1000./fps;
            e2e_delay[cnt-cycle_offset] = display_end - frame[display_index].frame_timestamp;
            e_disp_array[cnt-cycle_offset] = d_disp - b_disp;
            b_disp_array[cnt-cycle_offset] = b_disp;
            d_disp_array[cnt-cycle_offset] = d_disp;
            slack[cnt - cycle_offset] = slack_time;
            num_object_array[cnt - cycle_offset] = num_object;

            //printf("num_object : %d\n", num_object);
            //printf("slack: %f\n",slack[cnt-cycle_offset]);
            printf("latency: %f\n",e2e_delay[cnt-cycle_offset]);
            printf("cnt : %d\n",cnt);
        }
#ifdef ZERO_SLACK
        else if(cnt < (cycle_offset - 1))
        {
            printf("MIN : %f\n", MIN(s_min, (1000./fps)));
            printf("e_fetch_max : %f\n", MAX(e_fetch_max, e_fetch));
            printf("b_fetch_max : %f\n", MAX(b_fetch_max, b_fetch));
            s_min = MIN(s_min, (1000./fps));
            e_fetch_max = MAX(e_fetch_max, e_fetch);
            b_fetch_max = MAX(b_fetch_max, b_fetch);
        }
        else if(cnt == (cycle_offset - 1)){
            fetch_offset = (int)(s_min - e_fetch_max - b_fetch_max);
            printf("fetch_offset : %d\n", fetch_offset);
        }
#endif

        /* Exit object detection cycle */
        if(cnt == ((obj_det_cycle_idx + cycle_offset)-1)){
            int exist=0;
            FILE *fp;
            char file_path[100] = "";

            strcat(file_path, MEASUREMENT_PATH);
            strcat(file_path, MEASUREMENT_FILE);

            fp=fopen(file_path,"w+");

            if (fp == NULL) 
            {
                /* make directory */
                while(!exist){
                    int result;

                    result = mkdir(MEASUREMENT_PATH, 0766);

                    if(result == 0) { 
                        exist = 1;

                        fp=fopen(file_path,"w+");
                    }
                }
            }
            fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", "e_fetch", "b_fetch", "d_fetch",
                    "e_infer", "b_infer", "d_infer", "e_disp", "b_disp", "d_disp",
                    "slack", "e2e_delay", "fps", "c_sys", "IFG", "n_obj");

            for(int i=0;i<obj_det_cycle_idx;i++){
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
                        e2e_delay[i], slack[i], fps_array[i], cycle_time_array[i], inter_frame_gap_array[i], num_object_array[i]);
            }
            fclose(fp);

            /* exit loop */
            break;
        }
        cnt++;
        buff_index = (buff_index + 1) % 3;
        printf("================end===============\n");
    }
    cnt = 0;

    /* Average data */
    printf("============ Darknet data ===========\n");
    printf("Avg fetch execution time (ms) : %0.2f\n", e_fetch_sum / obj_det_cycle_idx);
    printf("Avg fetch blocking time (ms) : %0.2f\n", b_fetch_sum / obj_det_cycle_idx);
    printf("Avg fetch delay (ms) : %0.2f\n", d_fetch_sum / obj_det_cycle_idx);
    printf("Avg infer execution on cpu (ms) : %0.2f\n", e_infer_cpu_sum / obj_det_cycle_idx);
    printf("Avg infer execution on gpu (ms) : %0.2f\n", e_infer_gpu_sum / obj_det_cycle_idx);
    printf("Avg infer delay (ms) : %0.2f\n", d_infer_sum / obj_det_cycle_idx);
    printf("Avg disp execution time (ms) : %0.2f\n", e_disp_sum / obj_det_cycle_idx);
    printf("Avg disp blocking time (ms) : %0.2f\n", b_disp_sum / obj_det_cycle_idx);
    printf("Avg disp delay (ms) : %0.2f\n", d_disp_sum / obj_det_cycle_idx);
    printf("Avg salck (ms) : %0.2f\n", slack_sum / obj_det_cycle_idx);
    printf("Avg E2E delay (ms) : %0.2f\n", e2e_delay_sum / obj_det_cycle_idx);
    printf("Avg cycle time (ms) : %0.2f\n", cycle_time_sum / obj_det_cycle_idx);
    printf("Avg inter frame gap : %0.2f\n", inter_frame_gap_sum / obj_det_cycle_idx);
    printf("Avg number of object : %0.2f\n", num_object_sum / obj_det_cycle_idx);
    printf("=====================================\n");

    printf("input video stream closed. \n");
    if (output_video_writer) {
        release_video_writer(&output_video_writer);
        printf("output_video_writer closed. \n");
    }

    // free memory
    //printf("1\n");
    free_image(in_s);
    //printf("2\n");
    free_detections(dets, nboxes);
    //printf("3\n");

    free(avg);
    //printf("3.1\n");
    for (j = 0; j < NFRAMES; ++j) free(predictions[j]);
    demo_index = (NFRAMES + demo_index - 1) % NFRAMES;
    //printf("3.2\n");
    for (j = 0; j < NFRAMES; ++j) {
        release_mat(&cv_images[j]);
        printf("j = %d\n", j);
        printf("NAFRAMES = %d\n", NFRAMES);
    }
    printf("4\n");
    free_ptrs((void **)names, net.layers[net.n - 1].classes);
    printf("5\n");
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
void r_tod(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, char **names, int classes,
        int frame_skip, char *prefix, char *out_filename, int mjpeg_port, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, char *http_post_host,
        int benchmark, int benchmark_layers)
{
    fprintf(stderr, "R-TOD needs OpenCV for webcam images.\n");
}
#endif
