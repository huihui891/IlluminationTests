#ifndef ILLUMINATIONTESTS_H
#include "IlluminationTests.h"
#endif

class LoggingUtils
{
public:
	static void AddToImageGrid(cv::Mat image, std::string caption);
	static cv::Mat RenderGrid(Size s);
	static cv::Mat plot_histogram(const cv::Mat& img);
private:
	static std::vector<cv::Mat> grid_images;
};