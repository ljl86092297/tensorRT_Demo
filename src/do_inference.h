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

	//ͨ����ȡonnx�ļ�ת��ΪtensorRT������ģ�Ͳ���������
	bool infer();
	bool build();
	//ͨ��ֱ�Ӷ�ȡtensorRT�ļ�������ģ�Ͳ���������
	bool read_build();
	bool read_infer();
	bool read_post(); //��Ԥ�����ݽ��д������ӻ���ͼƬ��


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


	//�˷���ͨ�����鷽����������
	bool read_processInput(float*& input, const std::string& path);

	//�˷���ֱ��ͨ��cv::dn���ú���
	bool read_processInput(float*& input,const bool flag);

};

