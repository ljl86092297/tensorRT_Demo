#pragma once

#include <NvInfer.h>
#include "parserOnnxConfig.h"
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

	//通过读取onnx文件转换为tensorRT来创建模型并进行推理
	bool infer();
	bool build();
	//通过直接读取tensorRT文件来创建模型并进行推理
	bool read_build();
	bool read_infer();
	bool read_post(); //对预测数据进行处理并可视化到图片中


	bool verifyOutput(float* a,const  cv::Size& inputFrameSize, const cv::Size& originaleFrameSize, const float& conf, const float& iou);

	bool doMain();

	~yoloRT()
	{
		delete trtModelStream;
		delete prob;
	}

private:
	SampleParams mParams;
	nvinfer1::Dims mInputDims;
	nvinfer1::Dims mOutputDims;

	std::shared_ptr<nvinfer1::ICudaEngine>  mEngine;
	IExecutionContext* context;
	cv::Mat ori_img;
	std::vector<Detection> detects;
	cv::Size oriImageSize;

	size_t model_size;
	std::string inputName;
	std::string outputName;


	char* trtModelStream{ nullptr };
	float* data{ nullptr };
	float* prob{ nullptr };

	bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
		SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser);
	bool processInput(const samplesCommon::BufferManager& buffer, cv::Mat& frame);
	
	bool getMaxclsConf(std::vector<float>::iterator it, const int& numCls, float& maxclsConf, int& bestclsId);
	
	bool scaleBox2OriSize(const cv::Size& inputFrameSize, const cv::Size& originaleFrameSize, cv::Rect& box);


	//此方法通过数组方法操作数据
	bool read_processInput(float*& input, const std::string& path);

	//此方法直接通过cv::dn内置函数
	bool read_processInput(float*& input,const bool flag);

};

