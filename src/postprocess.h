#pragma once

#include "NvInfer.h"
#include "baseStruct.h"



class PostProcess
{
public:

	PostProcess(const cv::Size& originalFrameSize, const cv::Size& inputFrameSize = cv::Size(640, 640), const float& conf = confThreshold, const float& iou = iouThreshold)
		:selforiginalFrameSize(originalFrameSize), selfinputFrameSize(inputFrameSize), selfconf(conf), selfiou(iou) {};

	bool getMaxclsConf(std::vector<float>::iterator it, const int& numCls, float& maxclsConf, int& bestclsId);

	bool scaleBox2OriSize(cv::Rect& box);

	bool outputs2dets(float* prob, const nvinfer1::Dims& outputdims);

	const std::vector<Detection>& getDetects()const { return detects; };
private:
	std::vector<Detection> detects;
	cv::Size selfinputFrameSize;
	cv::Size selforiginalFrameSize;
	float selfconf;
	float selfiou;
	

};