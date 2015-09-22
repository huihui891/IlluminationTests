#include "LoggingUtils.h"

using namespace cv;
using namespace std;

std::vector<Mat> LoggingUtils::grid_images;

void LoggingUtils::AddToImageGrid(Mat image, String caption){
	/* Adds caption to image and image to the grid */
	Mat tmp = image.clone();
	putText(tmp, caption, Point(20, 20), FONT_HERSHEY_PLAIN, 2.0, CV_RGB(255, 0, 0), 2.0);
	grid_images.push_back(tmp);
}
Mat LoggingUtils::RenderGrid(Size s){
	/* Create and render image grid */
	Mat grid(s.height, s.width, CV_8UC3, cv::Scalar(0, 0, 0));

	int num_images = LoggingUtils::grid_images.size();
	int grid_x = (int) ceil(sqrt(num_images));
	int grid_y = (int) ceil((double) num_images / grid_x);

	int width = grid.cols / grid_x;
	int height = grid.rows / grid_y;

	int k = 0;
	for (int i = 0; i < grid_y; i++){
		for (int j = 0; j < grid_x; j++){
			if (k == num_images){
				LoggingUtils::grid_images.clear();
				return grid;
			}
			Mat tmp = LoggingUtils::grid_images[k++];
			assert(tmp.type() == CV_8UC3);
			resize(tmp, tmp, Size(width, height));
			tmp.copyTo(grid(Rect(j*width, i*height, width, height)));
		}
	}
	LoggingUtils::grid_images.clear();
	return grid;
}
Mat LoggingUtils::plot_histogram(const Mat& img){
	/* Plots histogram of input image and returns */
	int hist_w = 512; int hist_h = 400;
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true, accumulate = false;

	switch (img.channels()) {
	case 1:{
			int bin_w = cvRound((double)hist_w / histSize);
			Mat hist;
			calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
			normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
			for (int i = 1; i < histSize; i++) {
				line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
					Point(bin_w*i, hist_h - cvRound(hist.at<float>(i))),
					Scalar(255, 0, 0), 2, 8, 0);
			}
		}
		break;
	case 3:{
			//cout << "Plotting histogram for 3 channel image" << endl;
			/// Separate the image in 3 places ( B, G and R )
			vector<Mat> bgr_planes;
			split(img, bgr_planes);

			Mat b_hist, g_hist, r_hist;

			/// Compute the histograms:
			calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
			calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
			calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

			// Draw the histograms for B, G and R
			int bin_w = cvRound((double)hist_w / histSize);

			/// Normalize the result to [ 0, histImage.rows ]
			normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
			normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
			normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

			/// Draw for each channel
			for (int i = 1; i < histSize; i++)
			{
				line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
					Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
					Scalar(255, 0, 0), 2, 8, 0);
				line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
					Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
					Scalar(0, 255, 0), 2, 8, 0);
				line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
					Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
					Scalar(0, 0, 255), 2, 8, 0);
			}
		}
		break;
	default:
		break;
	}
	return histImage;
}