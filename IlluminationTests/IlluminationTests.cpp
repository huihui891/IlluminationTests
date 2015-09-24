#ifndef ILLUMINATIONTESTS_H
#include "IlluminationTests.h"
#endif
#include "LoggingUtils.h"

using namespace std;
using namespace cv;
using cv::CLAHE;


Mat tan_triggs_preprocessing(Mat& current_image,
	float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1,
	int sigma1 = 2) {
	/* Implementation of INRIA Tan Paper */
	
	/* 1. Gamma 'Correction' */
	Mat X, I;
	current_image.convertTo(X, CV_32FC1);
	pow(X, gamma, I);

	/* 2. Difference of Gaussians */
	Mat gaussian0, gaussian1;
	int kernel_sz0 = (3 * sigma0);
	int kernel_sz1 = (3 * sigma1);

	kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
	kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
	GaussianBlur(I, gaussian0, Size(kernel_sz0, kernel_sz0), sigma0, sigma0, BORDER_REPLICATE);
	GaussianBlur(I, gaussian1, Size(kernel_sz1, kernel_sz1), sigma1, sigma1, BORDER_REPLICATE);
	subtract(gaussian0, gaussian1, I);
	
	double meanI = 0.0;

	Mat tmp;
	pow(abs(I), alpha, tmp);
	meanI = mean(tmp).val[0];

	I = I / pow(meanI, 1.0 / alpha);

	meanI = 0.0;

	pow(min(abs(I), tau), alpha, tmp);
	meanI = mean(tmp).val[0];

	I = I / pow(meanI, 1.0 / alpha);

	Mat exp_x, exp_negx;
	exp(I / tau, exp_x);
	exp(-I / tau, exp_negx);
	divide(exp_x - exp_negx, exp_x + exp_negx, I);
	I = tau * I;
	I.convertTo(I, CV_8UC1);
	return I;
}
Mat norm_0_255(const Mat&src){
	/* 3. Contrast equalization */
	Mat dst;
	switch (src.channels()) {
	case 1:
		normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}
Mat tan_triggs(Mat& current_image){
	Mat current_tan_final;
	vector<Mat> channels_tan;

	cvtColor(current_image, current_image, CV_BGR2Lab);
	split(current_image, channels_tan);
	Mat current_preprocessed = tan_triggs_preprocessing(channels_tan[0]);
	current_preprocessed = norm_0_255(current_preprocessed);
	current_preprocessed.copyTo(channels_tan[0]);
	merge(channels_tan, current_tan_final);
	cvtColor(current_tan_final, current_tan_final, CV_Lab2BGR);

	return current_tan_final;
}
Mat DOG(Mat& I, int sigma0 = 1, int sigma1 = 2){
	Mat gaussian0, gaussian1, current_DOG;
	int kernel_sz0 = (3 * sigma0);
	int kernel_sz1 = (3 * sigma1);

	kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
	kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
	GaussianBlur(I, gaussian0, Size(kernel_sz0, kernel_sz0), sigma0, sigma0, BORDER_REPLICATE);
	GaussianBlur(I, gaussian1, Size(kernel_sz1, kernel_sz1), sigma1, sigma1, BORDER_REPLICATE);
	subtract(gaussian0, gaussian1, I);

	switch (I.channels()) {
	case 1:
		normalize(I, current_DOG, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		normalize(I, current_DOG, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		I.copyTo(current_DOG);
		break;
	}
	return current_DOG;
}
Mat gamma_decorrection(Mat& current_image, double gamma){
	/* Gamma Correction */
	Mat current_gamma;
	Mat lut_matrix(1, 256, CV_8UC1);
	uchar *ptr = lut_matrix.ptr();
	//double inverse_gamma = 1.0 / gamma;

	for (int i = 0; i < 256; i++)
		ptr[i] = (int)(pow((double)i / 255.0, gamma) * 255.0);
	LUT(current_image, lut_matrix, current_gamma);
	return current_gamma;
}

Mat clahe_transformation(Mat& current_image, int clip_limit = 1, int tile_size = 8){
	/* CLAHE */
	Mat current_clahe;
	vector<Mat> channels_clahe;
	
	cvtColor(current_image, current_image, CV_BGR2Lab);
	split(current_image, channels_clahe);

	Mat plot_clahe = LoggingUtils::plot_histogram(channels_clahe[0]);
	LoggingUtils::AddToImageGrid(plot_clahe, "Input Histogram CLAHE");

	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(clip_limit);
	if (tile_size != 8)
		clahe->setTilesGridSize(cv::Size(tile_size, tile_size));

	clahe->apply(channels_clahe[0], channels_clahe[0]);
	
	plot_clahe = LoggingUtils::plot_histogram(channels_clahe[0]);
	LoggingUtils::AddToImageGrid(plot_clahe, "Output Histogram CLAHE");

	merge(channels_clahe, current_clahe);
	cvtColor(current_clahe, current_clahe, CV_Lab2BGR);

	LoggingUtils::AddToImageGrid(current_clahe, "CLAHE Output");

	return current_clahe;
}
Mat histeq_transformation(Mat& current_image){
	/* Histogram Equalization */
	Mat current_histeq;
	vector<Mat> channels_histeq;
	cvtColor(current_image, current_image, CV_BGR2Lab);
	split(current_image, channels_histeq);
	
	Mat plot_histeq = LoggingUtils::plot_histogram(channels_histeq[0]);
	LoggingUtils::AddToImageGrid(plot_histeq, "Input Histogram HistEq");

	equalizeHist(channels_histeq[0], channels_histeq[0]);

	plot_histeq = LoggingUtils::plot_histogram(channels_histeq[0]);

	LoggingUtils::AddToImageGrid(plot_histeq, "Output Histogram HistEq");

	merge(channels_histeq, current_histeq);
	cvtColor(current_histeq, current_histeq, CV_Lab2BGR);

	LoggingUtils::AddToImageGrid(current_histeq, "HistEq Output");

	return current_histeq;
}

Mat bilateral_filtering(Mat& current_image){
	Mat dst_image;
	for (int i = 1; i <= 6; i = i + 2){
		bilateralFilter(current_image, dst_image, i, i * 2, i / 2);
		current_image = dst_image.clone();
	}
	return current_image;
}

int main(){

	cout << "Enter filename (relative to data/):" << endl;
	string fileName;
	cin >> fileName;
	string videoFile = "C:/Users/virprabh/Documents/Visual Studio 2013/Projects/IlluminationTests/data/" + fileName;
	VideoCapture capture = videoFile.empty() ? VideoCapture(0) : VideoCapture(videoFile);
	if (!capture.isOpened())
	{
		cout << "Video capture could not be opened." << endl;
		return -1;
	}
	Mat current_image;
	Mat current_clahe, current_histeq;
	Mat current_transformation;
	int fwidth = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int fheight = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	cv::Size frameSize(fwidth, fheight);
	string outputFilePath = "C:/Users/virprabh/Documents/Visual Studio 2013/Projects/IlluminationTests/Results/" + fileName;
	int fps = 10;
	int fourCCforFile = 844516695;
	int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));
	VideoWriter writer(outputFilePath, fourCCforFile, fps, frameSize, true);
	if (!writer.isOpened())
	{
		cout << "Video writer could not be opened." << endl;
		return -1;
	}
	for (;;){

		if ( !capture.read(current_image) ){
			cout << "Stream end" << endl;
			break;	
		}

		LoggingUtils::AddToImageGrid(current_image, "Original Stream");

		/* Illumination Normalization Experiments */
		
		//Mat current_tan_final = tan_triggs(current_image.clone());
		//LoggingUtils::AddToImageGrid(current_tan_final, "Tan Final");
		
		double gamma = 2.2;
		Mat current_gamma = gamma_decorrection(current_image.clone(), gamma);
		LoggingUtils::AddToImageGrid(current_gamma, "Current Gamma");

		cvtColor(current_gamma, current_gamma, CV_BGR2Lab);
		Mat plot_gamma = LoggingUtils::plot_histogram(current_gamma);

		LoggingUtils::AddToImageGrid(plot_gamma, "Gamma Histogram");

		Mat current_clahe = clahe_transformation(current_image.clone());
		Mat current_histeq = histeq_transformation(current_image.clone());
		
		Mat current_DOG = DOG(current_image.clone());
		LoggingUtils::AddToImageGrid(current_DOG, "Current DOG");

		//cvtColor(current_DOG, current_DOG, CV_BGR2Lab);
		//Mat plot_dog = LoggingUtils::plot_histogram(current_DOG);

		//LoggingUtils::AddToImageGrid(plot_dog, "DOG Histogram");

		/* End of Illumination Normalization Experiments */

		//Mat current_bilateral = bilateral_filtering(current_image);

		/* End of experiments */
		Mat output = LoggingUtils::RenderGrid(Size(fwidth, fheight));
		const Mat final = output.clone(); 
		imshow("Grid", final);
		assert(final.size() == current_image.size());
		assert(final.type() == CV_8UC3);
		writer.write(final);
		cv::waitKey(30);
	}
	capture.release();
	//writer.release();
}