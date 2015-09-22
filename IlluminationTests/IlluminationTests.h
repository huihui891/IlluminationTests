#define ILLUMINATIONTESTS_H
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

Mat tan_triggs_preprocessing(Mat& current_image, float alpha, float tau, 
	float gamma, int sigma0, int sigma1);
Mat norm_0_255(const Mat&src);
Mat tan_triggs(Mat& current_image);
Mat DOG(Mat& I, int sigma0, int sigma1);
Mat gamma_correction(Mat& current_image, double gamma);
Mat clahe_transformation(Mat& current_image, int clip_limit, int tile_size);
Mat histeq_transformation(Mat& current_image);
Mat bilateral_filtering(Mat& current_image);