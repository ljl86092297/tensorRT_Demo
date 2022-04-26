#pragma once

#include <NvInfer.h>
#include "common.h"
#include "buffers.h"
#include "preprocess.h"
#include "argsParser.h"

#include <cuda_runtime_api.h>
#include "baseStruct.h"
using samplesCommon::SampleUniquePtr;






class yoloRT
{

public:
	yoloRT(const SampleParams& params)
		:mParams(params), mEngine(nullptr) {};

	bool read_build();
	bool read_infer();

	bool verifyOutput();//对预测数据进行处理并可视化到图片中

	bool doMain();

	~yoloRT()
	{
		delete trtModelStream;
		delete prob;
	}

private:
	SampleParams mParams;


	std::shared_ptr<nvinfer1::ICudaEngine>  mEngine;
	IExecutionContext* context;
	cv::Mat ori_img;
	std::vector<Detection> detects;
	cv::Size oriImageSize;
	nvinfer1::Dims mOutputDims;
	size_t model_size;
	std::string inputName;
	std::string outputName;


	char* trtModelStream{ nullptr };
	float* data{ nullptr };
	float* prob{ nullptr };

	//此方法通过数组方法操作数据
	bool read_processInput(float*& input, const std::string& path);

	//此方法直接通过cv::dn内置函数
	bool read_processInput(float*& input,const bool flag);

};

