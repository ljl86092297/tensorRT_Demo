#include "onnx2trt.h"
#include "preprocess.h"
#include "image.h"

bool Onnx2Trt::build()
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
	if (!parser)
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
	/*modelStream->destroy();*/
	p.close();
	mOutputDims = network->getOutput(0)->getDimensions();

	ASSERT(network->getNbInputs() == 1);
	inputName = network->getInput(0)->getName();
	outputName = network->getOutput(0)->getName();

	return true;
}

bool Onnx2Trt::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder, SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config, SampleUniquePtr<nvonnxparser::IParser>& parser)
{

	string s = mParams.loadOnnx;
	auto parsed = parser->parseFromFile(s.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));

	if (!parsed)
	{
		std::cout << " parase onnx fail" << std::endl;
		return false;
	}

	if (mParams.fp16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	if (mParams.int8)
	{
		//int8 可能不行 我是通过python转换的 有需要代码的可以联系我
		config->setFlag(BuilderFlag::kINT8);
		samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
	}
	samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

	return true;
}


bool Onnx2Trt::processInput(const samplesCommon::BufferManager& buffers, cv::Mat& frame)
{
	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
	//hostDataBuffer = (float*)(frame.data);
	int i = 0;
	for (int row = 0; row < 640; ++row) {
		uchar* uc_pixel = frame.data + row * frame.step;
		for (int col = 0; col < 640; ++col) {
			hostDataBuffer[i] = (float)uc_pixel[2] / 255.0;
			hostDataBuffer[i + 640 * 640] = (float)uc_pixel[1] / 255.0;
			hostDataBuffer[i + 2 * 640 * 640] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;

		}
	}
	return true;
}

bool Onnx2Trt::verifyOutput()
{
	PostProcess post(oriImageSize, cv::Size(INPUT_H, INPUT_W));
	post.outputs2dets(prob, mOutputDims);
	detects = post.getDetects();

	return true;
}

bool Onnx2Trt::infer()
{
	//cv::Mat ori_img = cv::imread(mParams.imgRpath);
	oriImageSize = ori_img.size();
	std::cout << "oriImageSize   " << oriImageSize << std::endl;
	Preprocess p;
	cv::Mat pre_img = p.preprocess_img(ori_img);
	samplesCommon::BufferManager buffers(mEngine);
	auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (!context)
	{
		std::cout << " creat context fail" << std::endl;
		return false;
	}
	mParams.outputTensorNames.push_back(outputName);
	mParams.inputTensorNames.push_back(inputName);
	ASSERT(mParams.inputTensorNames.size() == 1);
	std::cout << "pre_img size:  " << pre_img.size << std::endl;
	if (!processInput(buffers, pre_img))
	{
		std::cout << " buffers get data fail" << std::endl;
		return false;
	}
	buffers.copyInputToDevice();

	bool status = context->executeV2(buffers.getDeviceBindings().data());
	if (!status)
	{
		std::cout << " inference fail" << std::endl;
		return false;
	}
	std::cout << "infer sucess" << std::endl;
	buffers.copyOutputToHost();

	prob = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
	if (!verifyOutput())
	{
		std::cout << "output problem" << std::endl;
	}
	return true;
}


bool Onnx2Trt::doMain()
{
	Image img(mParams.imgRpath);
	ori_img = img.get_oriImage();
	if (!build())
	{
		std::cout << "model build defail" << std::endl;
		return false;
	}
	if (!infer())
	{
		std::cout << "model infer defail" << std::endl;

		return false;
	}
	if (mParams.savef)
	{
		img.draw(detects, mParams.imgWpath);
		return true;
	}

	return true;
}
