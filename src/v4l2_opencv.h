#include "darknet.h"

using namespace cv;

extern "C" {

Mat image_to_mat(image im);
image ipl_to_image(IplImage* src);
image mat_to_image(Mat m);

}
