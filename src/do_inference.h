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

	bool verifyOutput();//��Ԥ�����ݽ��д������ӻ���ͼƬ��

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

	//�˷���ͨ�����鷽����������
	bool read_processInput(float*& input, const std::string& path);

	//�˷���ֱ��ͨ��cv::dn���ú���
	bool read_processInput(float*& input,const bool flag);

};

