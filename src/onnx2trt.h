#pragma once
#include "baseStruct.h"
#include <NvInfer.h>
#include "parserOnnxConfig.h"
#include "buffers.h"
#include <common.h>
#include "postprocess.h"
using samplesCommon::SampleUniquePtr;


class Onnx2Trt
{
public:
	Onnx2Trt(const SampleParams& params)
		:mParams(params), mEngine(nullptr) {};
	bool infer();
	bool build();
	bool doMain();


private:
	SampleParams mParams;
	cv::Size oriImageSize;
	std::string inputName;
	std::string outputName;
	std::vector<Detection> detects;
	nvinfer1::Dims mOutputDims;
	std::shared_ptr<nvinfer1::ICudaEngine>  mEngine;
	cv::Mat ori_img;
	float* prob{ nullptr };

	bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
		SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser);
	bool processInput(const samplesCommon::BufferManager& buffer, cv::Mat& frame);

	bool verifyOutput();



};