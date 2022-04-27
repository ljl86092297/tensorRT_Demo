#include "do_inference.h"
#include "postprocess.h"


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
	//std::cout << "loading img and process img success hava flag" << std::endl;
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
	//std::cout << "init getBinding" << std::endl;

	CHECK(cudaMalloc(&buffers[0], mOutputDims.d[0] * 3 * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[1], 1 * 100 * sizeof(float)));
	CHECK(cudaMalloc(&buffers[2], 1 * 100 * sizeof(float)));
	CHECK(cudaMalloc(&buffers[3], 1 * 100 * sizeof(float)));
	CHECK(cudaMalloc(&buffers[4], mOutputDims.d[0] * mOutputDims.d[1] * mOutputDims.d[2] * sizeof(float)));

	//std::cout << "malloc cuda buffers success" << std::endl;
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	CHECK(cudaMemcpyAsync(buffers[0], data, mOutputDims.d[0] * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	//std::cout << "malloc cuda input data success2" << std::endl;
	context->enqueue(1, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(prob, buffers[4], mOutputDims.d[0]  *mOutputDims.d[1] * mOutputDims.d[2] * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
	for (int i = 0; i < 5; i++)
	{
		CHECK(cudaFree(buffers[i]));
	}
	//std::cout << "inference sucess" << std::endl;
	return true;
}


bool yoloRT::verifyOutput()
{
	PostProcess post(oriImageSize, cv::Size(INPUT_H, INPUT_W));
	post.outputs2dets(prob, mOutputDims);
	detects = post.getDetects();
	
	return true;
}

bool yoloRT::doMain()
{

	if (!read_build())
	{
		std::cout << "model build defail" << std::endl;
		return false;
	}
	if (mParams.videoRFlag)
	{
		flag = true;
		v = new  Video(mParams.videoRurl);
		if (mParams.videoSFlag)
		{
			v->create_videoMemory(mParams.videoWurl, 25);
		}
		
	}
	else
	{
		img = new Image(mParams.imgRpath);
		
	}

	while (1)
	{
		if (mParams.videoRFlag)
		{
			
			if (!v->get_frame(ori_img))
			{
				v->release();

				break;
			}
		}
		else
		{
			ori_img = img->get_oriImage();
		}
		if (!read_infer())
		{
			std::cout << "model infer defail" << std::endl;

			return false;
		}
		if (!verifyOutput())
		{
			std::cout << "parser output or draw image defail" << std::endl;
			return false;
		}
		if (mParams.videoRFlag)
		{
			v->draw(ori_img, detects);
			v->write_video(ori_img);
		}
		else
		{
			img->draw(detects, mParams.imgWpath);

		}

		if (!flag)
		{
			break;
		}
	}



	return true;
}

