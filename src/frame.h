#pragma once
#include<opencv2/opencv.hpp>
#include"baseStruct.h"

class Frame
{
public:
	Frame(const std::string imgpath) : imgPath(imgpath) {
		std::cout << "InitFrame Object Sucess" << std::endl;
		srand(time(0));
		for (int i = 0; i < 80; i++) {
			int b = rand() % 256;
			int g = rand() % 256;
			int r = rand() % 256;
			color.push_back(cv::Scalar(b, g, r));
		}
	};
	bool draw(cv::Mat& frame,const std::vector<Detection>& dets);
	bool writeImg(cv::Mat& frame);
	bool wirteVideo();
private:
	std::vector<cv::Scalar> color;
	std::string imgPath;
};