#include "do_inference.h"
#include "postprocess.h"

#include "image.h"
#include <fstream>



bool yoloRT::verifyOutput(float* a, const cv::Size& inputFrameSize, const  cv::Size& originaleFrameSize, const float& conf = confThreshold, const float& iou = iouThreshold)
{
	size_t nc = mOutputDims.d[2] - 5;
	size_t count = mOutputDims.d[0] * mOutputDims.d[1] * mOutputDims.d[2];
	std::vector<float> output(a, a + count);
	std::cout << output.size() << std::endl;

	std::vector<cv::Rect> boxs;
	std::vector<int> clsIds;
	std::vector<float>confs;
	int elementsInBatch = (int)(mOutputDims.d[1] * mOutputDims.d[2]);

	for (auto it = output.begin(); it != output.end(); it += mOutputDims.d[2])
	{
		float objConf = it[4];
		if (objConf > conf)
		{
			int centerX = it[0];
			int centerY = it[1];
			int width = it[2];
			int height = it[3];
			int left = centerX - width / 2;
			int top = centerY - height / 2;

			float maxclsConf;
			int bestclsId;

			getMaxclsConf(it, nc, maxclsConf, bestclsId);

			// co_conf  is class and object ;
			float co_conf = maxclsConf * objConf;

			confs.emplace_back(co_conf);
			clsIds.emplace_back(bestclsId);
			boxs.emplace_back(left, top, width, height);

		}
	}
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxs, confs, conf, iou, indices);
	for (int idx : indices)
	{
		std::cout << "is idx is ok" << std::endl;
		Detection det;
		det.box = cv::Rect(boxs[idx]);
		det.conf = confs[idx];
		det.clsId = clsIds[idx];

		scaleBox2OriSize(inputFrameSize, oriImageSize, det.box);
		detects.emplace_back(det);
	}

	return true;
}

bool yoloRT::build()
{
	auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));

	if (!builder)
	{
		return false;
	}

	const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
	if (!network)
	{
		return false;
	}

	auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

	if (!config)
	{
		return false;
	}

	auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
	if(!parser)
	{
		return false;
	}

	auto constructed = constructNetwork(builder, network, config, parser);
	if (!constructed)
	{
		std::cout << " 创建constructed失败" << std::endl;
		return false;
	}
	builder->setMaxBatchSize(16);

	auto profileStream = samplesCommon::makeCudaStream();
	if (!profileStream)
	{
		return false;
	}
	config->setProfileStream(*profileStream);
	
	SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };

	if (!plan)
	{
		std::cout << " 创建plan失败" << std::endl;
		return false;
	}

	SampleUniquePtr<IRuntime> runtime{ createInferRuntime(sample::gLogger.getTRTLogger()) };
	if (!runtime)
	{
		return false;
	}

	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());

	if (!mEngine)
	{
		std::cout << " 创建mEngine失败" << std::endl;
		return false;
	}
	
	//序列化engine  并存储成文件。
	IHostMemory* modelStream{ nullptr };
	modelStream = mEngine->serialize();
	std::ofstream p(mParams.loadEngine, std::ios::binary);

	p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
	//modelStream->destroy();
	p.close();


	ASSERT(network->getNbInputs() == 1);
	mInputDims = network->getInput(0)->getDimensions();
	ASSERT(mInputDims.nbDims == 4);
	std::cout << "network->getNbOutputs():  " << network->getNbOutputs() << std::endl;
	//ASSERT(network->getNbOutputs() == 1);
	mOutputDims = network->getOutput(0)->getDimensions();
	inputName = network->getInput(0)->getName();
	outputName = network->getOutput(0)->getName();

	
	std::cout <<"mOutputDims:  " <<mOutputDims << std::endl;
	std::cout << "inputName :" << inputName << std::endl;
	std::cout << "outputName :" << outputName << std::endl;

	

	return true;
}

bool yoloRT::read_processInput(float*& input, const std::string& path)
{
	input = new float[INPUT_H * INPUT_W * 3];
	ori_img = cv::imread(path);
	oriImageSize = ori_img.size();
	std::cout << "oriImageSize   " << oriImageSize << std::endl;
	Preprocess p;
	cv::Mat pre_img = p.preprocess_img(ori_img);
	int i = 0;
	for (int row = 0; row < INPUT_H; ++row) {
		uchar* uc_pixel = pre_img.data + row * pre_img.step;
		for (int col = 0; col < INPUT_W; ++col) {
			input[i] = (float)uc_pixel[2] / 255.0;
			input[i + INPUT_W * INPUT_H] = (float)uc_pixel[1] / 255.0;
			input[i + 2 * INPUT_W * INPUT_H] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;

		}
	}
	std::cout << "loadimg success" << std::endl;
	return true;
}

bool yoloRT::read_processInput(float*& input, const bool flag)
{
	oriImageSize = ori_img.size();
	Preprocess p;
	cv::Mat pre_img = p.preAll(ori_img, cv::Size(INPUT_H, INPUT_W));
	input = reinterpret_cast<float*>(pre_img.data);
	std::cout << "loading img and process img success hava flag" << std::endl;
	return true;
}



//通过读取模型文件进行模型构建
bool yoloRT::read_build()
{
	
	std::ifstream file(mParams.loadEngine, std::ios::binary);
	//file.good()  返回true则流正常
	if (file.good())
	{
		file.seekg(0, file.end);
		model_size = file.tellg();
		file.seekg(0, file.beg);
		trtModelStream = new char[model_size];
		assert(trtModelStream);
		file.read(trtModelStream, model_size);
		file.close();
	}
	std::cout << "read file success" << std::endl;
	SampleUniquePtr<IRuntime> runtime{ createInferRuntime(sample::gLogger.getTRTLogger()) };
	assert(runtime != nullptr);
	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtModelStream, model_size));
	context = mEngine->createExecutionContext();
	std::cout << "create context success" << std::endl;
	return true;
}


bool yoloRT::read_infer()
{
	const int inputIndex = mEngine->getBindingIndex("images");
	const int outputIndex = mEngine->getBindingIndex("output");
	mOutputDims = mEngine->getBindingDimensions(outputIndex);

	assert(inputIndex == 0);
	assert(outputIndex == 4);

	prob  = new float[mOutputDims.d[0] * mOutputDims.d[1]*mOutputDims.d[2]];
	read_processInput(data, true);
	void* buffers[5];
	std::cout << "init getBinding" << std::endl;

	CHECK(cudaMalloc(&buffers[0], mOutputDims.d[0] * 3 * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[1], 1 * 100 * sizeof(float)));
	CHECK(cudaMalloc(&buffers[2], 1 * 100 * sizeof(float)));
	CHECK(cudaMalloc(&buffers[3], 1 * 100 * sizeof(float)));
	CHECK(cudaMalloc(&buffers[4], mOutputDims.d[0] * mOutputDims.d[1] * mOutputDims.d[2] * sizeof(float)));

	std::cout << "malloc cuda buffers success" << std::endl;
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	CHECK(cudaMemcpyAsync(buffers[0], data, mOutputDims.d[0] * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	std::cout << "malloc cuda input data success2" << std::endl;
	context->enqueue(1, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(prob, buffers[4], mOutputDims.d[0]  *mOutputDims.d[1] * mOutputDims.d[2] * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	for (int i = 0; i < 5; i++)
	{
		CHECK(cudaFree(buffers[i]));
	}
	std::cout << "inference sucess" << std::endl;
	return true;
}


bool yoloRT::read_post()
{
	PostProcess post(oriImageSize, cv::Size(INPUT_H, INPUT_W));
	post.outputs2dets(prob, mOutputDims);
	detects = post.getDetects();
	
	return true;
}

bool yoloRT::doMain()
{
	
	Image img(mParams.imgRpath);
	ori_img = img.get_oriImage();
	if (!read_build())
	{
		std::cout << "model build defail" << std::endl;
		return false;
	}
	if (!read_infer())
	{
		std::cout << "model infer defail" << std::endl;

		return false;
	}
	if (!read_post())
	{
		std::cout << "parser output or draw image defail" << std::endl;
		return false;
	}
	img.draw(detects, mParams.imgWpath);
	return true;
}


bool yoloRT::infer()
{
	cv::Mat ori_img = cv::imread(mParams.imgRpath);
	oriImageSize = ori_img.size();
	std::cout << "oriImageSize   " << oriImageSize << std::endl;
	Preprocess p;
	cv::Mat pre_img = p.preprocess_img(ori_img);
	samplesCommon::BufferManager buffers(mEngine);
	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (!context)
	{
		std::cout << " 创建context失败" << std::endl;
		return false;
	}
	mParams.outputTensorNames.push_back(outputName);
	mParams.inputTensorNames.push_back(inputName);
	ASSERT(mParams.inputTensorNames.size() == 1);
	std::cout << "pre_img size:  " << pre_img.size << std::endl;
	if (!processInput(buffers, pre_img))
	{
		std::cout << " buffers获取数据失败" << std::endl;
		return false;
	}
	buffers.copyInputToDevice();

	bool status = context->executeV2(buffers.getDeviceBindings().data());
	if (!status)
	{
		std::cout << " 推理失败" << std::endl;
		return false;
	}	
	std::cout << "infer sucess" << std::endl;
	buffers.copyOutputToHost();
	
	float* a = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
	if (!verifyOutput(a, inputsize, oriImageSize))
	{

	}
	//Frame fra("./bus_new.jpg");
	//fra.draw(ori_img, detect);
	////fra.writeImg(ori_img);
	return true;
}

bool yoloRT::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser)
{

	string s = "./yolov5.onnx";
	auto parsed = parser->parseFromFile(s.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));

	if (!parsed)
	{
		std::cout << " 解析onnx模型失败" << std::endl;
		return false;
	}

	if (mParams.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	if (mParams.int8)
	{
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
	}
	samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
		
	return true;
}




bool yoloRT::processInput(const samplesCommon::BufferManager& buffers, cv::Mat& frame)
{
	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
	//hostDataBuffer = (float*)(frame.data);

	int i = 0;
	for (int row = 0; row < 640; ++row) {
		uchar* uc_pixel = frame.data + row * frame.step;
		for (int col = 0; col < 640; ++col) {
			hostDataBuffer[i] = (float)uc_pixel[2]/255.0 ;
			hostDataBuffer[i + 640 * 640] = (float)uc_pixel[1] / 255.0;
			hostDataBuffer[i + 2 * 640 * 640] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
			
		}
	}
	return true;
}

bool yoloRT::getMaxclsConf(std::vector<float>::iterator it, const int& numCls, float& maxclsConf, int& bestclsId)
{
	maxclsConf = 0;
	bestclsId = 5;
	for (int i=5; i<numCls+5; i++)
	{
		if (it[i] > maxclsConf)
		{
			maxclsConf = it[i];
			bestclsId = i - 5;
		}
	}

	return true;
}

bool yoloRT::scaleBox2OriSize(const cv::Size& inputFrameSize, const cv::Size& originaleFrameSize, cv::Rect& box)
{
	float gain = std::min((float)inputFrameSize.width / (float)originaleFrameSize.width, (float)inputFrameSize.height / (float)originaleFrameSize.height);
	int pad[2] = { (int)(((float)inputFrameSize.width - (float)originaleFrameSize.width * gain) / 2.0f),
				(int)(((float)inputFrameSize.height - (float)originaleFrameSize.height * gain) / 2.0f) };

	box.x = (int)std::round((box.x - pad[0])/ gain);
	box.y = (int)std::round((box.y - pad[1]) / gain);
	box.width = (int)std::round(box.width / gain);
	box.height = (int)std::round(box.height / gain);

	return true;
}



