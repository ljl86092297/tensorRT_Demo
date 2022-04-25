#pragma once
#include<opencv2/opencv.hpp>
#include"baseStruct.h"

class Image
{
public:
	Image(const std::string imgpath) : imgPath(imgpath) {
		std::cout << "InitFrame Object Sucess" << std::endl;
		srand(time(0));
		for (int i = 0; i < 80; i++) {
			int b = rand() % 256;
			int g = rand() % 256;
			int r = rand() % 256;
			color.push_back(cv::Scalar(b, g, r));
		}
		selforiImage = cv::imread(imgPath);

	};
	float round(float src, int bits);
	bool draw(const std::vector<Detection>& dets, const std::string& wimgPath);
	bool writeImg(cv::Mat& image);
	bool wirteVideo();
	cv::Mat get_oriImage() const { return selforiImage; };

private:
	std::vector<cv::Scalar> color;
	std::string imgPath;
	cv::Mat selforiImage;


};