#include <iostream>
#include <fstream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

/*
	GLOBAL CONSTANTS
*/
const int FILTER_SIZE = 5;
const int IMAGE_SIZE = 225;
const int IMAGE_NUM = 4;
const int ITERATION_NUM = 1000;
const double RELATIVE_ERROR_MIN = 0.0001;
const double LEARNING_RATE = 0.01;

void train() {
	// storing original and noisy images
	std::vector<cv::Mat> imageSources, noisyImageSources;

	// create filter and initiate values as a mean filter
	cv::Mat filter = cv::Mat(FILTER_SIZE * FILTER_SIZE, 1, CV_64FC1, 1);

	// loading a bunch of images from files
	for (int i = 0; i < IMAGE_NUM; i++) {
		cv::Mat tmp = cv::imread("d:/opencv/images/image" + std::to_string(i + 1) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		imageSources.push_back(tmp.clone());
		noisyImageSources.push_back(tmp.clone());
	}



	// convert uchar-typed original images into double-type
	for (int i = 0; i < IMAGE_NUM; i++) {
		imageSources.at(i).convertTo(imageSources.at(i), CV_64FC1, 1 / 255.0);
		noisyImageSources.at(i).convertTo(noisyImageSources.at(i), CV_64FC1, 1 / 255.0);
	}

	// add noise gaussian noise to each image
	for (int i = 0; i < IMAGE_NUM; i++) {
		cv::Mat noise = cv::Mat(noisyImageSources.at(i).size(), CV_64FC1);
		cv::Mat result;
		cv::normalize(noisyImageSources.at(i), result, 0.0, 1.0, CV_MINMAX, CV_64FC1);
		cv::randn(noise, 0.1, 0.05);
		result = result + noise;
		cv::normalize(result, result, 0.0, 1.0, CV_MINMAX, CV_64FC1);
		result.convertTo(result, CV_64FC1, 1);
		noisyImageSources.at(i) = result.clone();
	}

	// add s & p noise to each image
	/*
	for (int i = 0; i < IMAGE_NUM; i++) {
		
		for (int row = 0; row < noisyImageSources.at(i).rows; row++) {
			for (int col = 0; col < noisyImageSources.at(i).cols; col++) {
				int rx = std::rand() % 30;
				int ry = std::rand() % 30;
				if (rx % 28 == 0) noisyImageSources.at(i).at<double>(row, col) = 0;
				if (ry % 29 == 0) noisyImageSources.at(i).at<double>(row, col) = 1;
			}
		}
	}*/


	// storing collection of images (vectorized, vertically).
	// this will be the ground truth to compare in gradient descent algorithm
	cv::Mat originalImageList(0, 1, CV_64FC1);

	// prepare ground truth
	for (int i = 0; i < IMAGE_NUM; i++) {
		cv::Mat tmp = imageSources.at(i).clone();
		tmp = tmp.reshape(1, IMAGE_SIZE * IMAGE_SIZE);

		cv::vconcat(originalImageList, tmp, originalImageList);
	}


	// storing patches of noisy images: (IMAGE_SIZE * IMAGE_SIZE * IMAGE_NUM) by (FILTER_SIZE * FILTER_SIZE) matrix
	cv::Mat imagePatchesList(IMAGE_SIZE * IMAGE_SIZE * IMAGE_NUM, FILTER_SIZE * FILTER_SIZE, CV_64FC1);

	// prepare noisy images dataset
	int xIndex = 0;
	int yIndex = 0;
	for (int imageIdx = 0; imageIdx < IMAGE_NUM; imageIdx++) {
		// create padded version of the image
		cv::Mat padded;
		cv::copyMakeBorder(noisyImageSources.at(imageIdx).clone(), padded, FILTER_SIZE / 2, FILTER_SIZE / 2, FILTER_SIZE / 2, FILTER_SIZE / 2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

		// patches vectorization
		for (int row = 0; row < noisyImageSources.at(imageIdx).rows; row++) {
			for (int col = 0; col < noisyImageSources.at(imageIdx).cols; col++) {
				xIndex = 0;

				for (int i = 0; i < FILTER_SIZE; i++) {
					for (int j = 0; j < FILTER_SIZE; j++) {
						imagePatchesList.at<double>(yIndex, xIndex) = padded.at<double>(row + i, col + j);
						xIndex++;
					}
				}
				yIndex++;
			}
		}
	}


	//== apply initial filter to get initial error
	cv::Mat filteredImage(imagePatchesList.rows, 1, CV_64FC1);
	filteredImage = imagePatchesList * filter;

	double error = cv::norm(filteredImage - originalImageList, cv::NORM_L2);
	double maxError = error;
	double curError = 0;
	double deltaError = std::abs(error - curError);

	std::cout << "Initial error: " << error << std::endl;


	/* ======================= BEGINNING OF GRADIENT DESCENT LOOP ======================= */

	// variables for visualization during iteration
	cv::Mat noisies;
	for (int i = 0; i < IMAGE_NUM; i++) noisies.push_back(noisyImageSources.at(i));
	cv::imshow("Ground truth", originalImageList.reshape(1, IMAGE_SIZE * IMAGE_NUM));
	cv::imshow("Noisy", noisies.reshape(1, IMAGE_SIZE * IMAGE_NUM));
	cv::Mat filterVis;

	// for storing error from all iterations
	std::ofstream errorList;
	errorList.open("errorList.csv");

	// for storing trained filter
	cv::FileStorage filterFile("filter.xml", cv::FileStorage::WRITE);

	// take note the time before iteration
	int64 t0 = cv::getTickCount();

	int iter = 0;
	while ((deltaError > RELATIVE_ERROR_MIN) && (iter < ITERATION_NUM)) {
		for (int f = 0; f < FILTER_SIZE * FILTER_SIZE; f++) {
			cv::Mat gradMat;
			cv::subtract(filteredImage, originalImageList, gradMat);
			cv::multiply(gradMat, imagePatchesList.col(f), gradMat);

			double gradient = 2 * cv::sum(gradMat)[0] / filteredImage.rows;

			//== update filter weights
			filter.at<double>(f, 0) = filter.at<double>(f, 0) - LEARNING_RATE * gradient;
		}

		// get filter response
		filteredImage = imagePatchesList * filter;

		// calculate error and relative error
		error = cv::norm(filteredImage - originalImageList, cv::NORM_L2);
		deltaError = std::abs(error - curError);

		// write error to file for first 70 iteration
		// (because the error will be somewhat saturated after 70th iteration)
		if (iter < 70) {
			errorList << std::to_string(iter + 1) << "," << std::to_string(error) << "\n";
		}


		// Print the error and delta error.	
		std::cout << "Error at iteration " << iter + 1 << " = " << error <<  std::endl;
		curError = error;

		// Uncomment to visualize updated filter and its response image
		filterVis = filter.clone().reshape(1, FILTER_SIZE);
		cv::resize(filterVis, filterVis, cv::Size(300, 300), 0, 0, cv::INTER_NEAREST);
		cv::imshow("Filter", filterVis);
		cv::imshow("Filtered Image", filteredImage.reshape(1, IMAGE_SIZE * IMAGE_NUM));
		cv::waitKey(1);


		iter++;
	}
	/*========================== END OF GRADIENT DESCENT LOOP =======================*/

	// get training duration
	int64 t1 = cv::getTickCount();
	double secs = (t1 - t0) / cv::getTickFrequency();
	std::cout << "Training time: " << secs << " seconds";


	// storing final filter to a file
	filterFile << "filter" << filter;


	// cleanups
	filter.release();
	errorList.close();

}



void test() {
	cv::Mat ori = cv::imread("d:/opencv/images/test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	ori.convertTo(ori, CV_64FC1, 1 / 255.0);
	cv::Mat testImage = ori.clone();

	// load filter values from file. The file contains the values of trained filter.
	cv::FileStorage filterFile("filter.xml", cv::FileStorage::READ);
	cv::Mat filter;
	filterFile["filter"] >> filter;
	filter = filter.reshape(1, 5);


	// gaussian noise
	cv::Mat noise = cv::Mat(testImage.size(), CV_64FC1);
	cv::Mat result;
	cv::normalize(testImage, result, 0.0, 1.0, CV_MINMAX, CV_64FC1);
	cv::randn(noise, 0.1, 0.1);
	result = result + noise;
	cv::normalize(result, result, 0.0, 1.0, CV_MINMAX, CV_64FC1);
	testImage = result.clone();

	// snp noise
	/*for (int row = 0; row < testImage.rows; row++) {
		for (int col = 0; col < testImage.cols; col++) {
			int rx = std::rand() % 30;
			int ry = std::rand() % 30;
			if (rx % 28 == 0) testImage.at<double>(row, col) = 0;
			if (ry % 29 == 0) testImage.at<double>(row, col) = 1;
		}
	}
	*/

	// pad test image
	cv::Mat tmp;
	cv::copyMakeBorder(testImage, tmp, FILTER_SIZE / 2, FILTER_SIZE / 2, FILTER_SIZE / 2, FILTER_SIZE / 2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

	// perform convolution
	cv::Mat out = testImage.clone();
	for (int row = 0; row < testImage.rows; row++) {
		for (int col = 0; col < testImage.cols; col++) {
			double sum = 0.0;
			for (int i = 0; i < FILTER_SIZE; i++) {
				for (int j = 0; j < FILTER_SIZE; j++) {
					sum += tmp.at<double>(row + i, col + j)*filter.at<double>(i, j);
				}
			}
			out.at<double>(row, col) = sum;
		}
	}


	cv::imshow("Test image", testImage);
	cv::imshow("Filtered test image", out);

	cv::Mat tmp1, tmp2;
	cv::subtract(testImage, ori, tmp1); // noisy - original
	cv::subtract(out, ori, tmp2); // filtered - original
	std::cout << "noisy - original: " << cv::norm(tmp1, cv::NORM_L2) << std::endl;
	std::cout << "(noisy*filter) - original: " << cv::norm(tmp2, cv::NORM_L2) << std::endl;

	cv::imshow("Original image", ori);
	cv::imshow("Noisy image", testImage);
	cv::imshow("Filter response", out);
	cv::waitKey();
}

void main() {
	train();
	test();
}