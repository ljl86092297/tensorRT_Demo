#pragma once

#include "baseStruct.h"

class Video
{
public:
	Video(const std::string& videoUrl);
	bool get_frame(cv::Mat& frame);
	bool create_videoMemory(const std::string& saveVideoPath, const int& fps);
	bool write_video(const cv::Mat& frame);
	bool release();

	bool draw(cv::Mat& frame, const std::vector<Detection>& dets);

private:
	cv::VideoCapture cap;
	cv::VideoWriter output;
	cv::Size saveSize;
	std::vector<cv::Scalar> color;

};