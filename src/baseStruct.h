#pragma once
#include<iostream>
#include<vector>
#include <string>
#include <opencv2/opencv.hpp>

const float confThreshold = 0.25f;
const float iouThreshold = 0.45f;
static const int INPUT_H = 640;
static const int INPUT_W = 640;
const cv::Size inputsize = cv::Size(INPUT_H, INPUT_W);



struct SampleParams
{
	int32_t	batchSize{ 1 };
	int32_t dlaCore{ -1 };
	bool int8{ false };
	bool fp16{ false };
	//int32_t inputSize{640};
	std::vector<std::string> inputTensorNames;
	std::vector<std::string> outputTensorNames;
	std::string imgRpath{"./bus.jpg"};
	std::string imgWpath{"./bus_new.jpg"};
	std::string videoRurl{""};
	std::string videoWurl{""};
	std::string loadEngine{"./yolov5s_c++.trt"};
	std::string saveEngine{""};


};

struct Detection
{
	cv::Rect box;
	float conf{};
	int clsId{};
};

const std::vector<std::string> clsName = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	   "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
	   "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	   "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	   "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
	   "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
	   "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
	   "hair drier", "toothbrush" };





