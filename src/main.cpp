#include "do_inference.h"
#include "logging.h"


SampleParams initializeSampleParams(const samplesCommon::Args& args)
{
	SampleParams params;
	if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
	{
		params.dataDirs.push_back("data/mnist/");
		params.dataDirs.push_back("data/samples/mnist/");
	}
	else // Use the data directory provided by the user
	{
		params.dataDirs = args.dataDirs;
	}
	//params.FileName = "yolov5s.onnx";

	params.dlaCore = -1;
	params.int8 = args.runInInt8;
	params.fp16 = args.runInFp16;

	params.wpath = "./new_bus.jpg";
	params.rpath = "./bus.jpg";
	std::cout << params.int8 << std::endl;
	std::cout << params.fp16 << std::endl;
	std::cout << args.useDLACore << std::endl;
	return params;

}




int main(int argc, char** argv)
{
	samplesCommon::Args args;
	bool argsOk = samplesCommon::parseArgs(args, argc, argv);
	if (!argsOk)
	{
		std::cout << "ÃüÁî½âÎöÊ§°Ü" << std::endl;
	}

	if (args.help)
	{
		return 0;
	}
	const std::string gSampleName = "TensorRT.sample_onnx_mnist";
	auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

	sample::gLogger.reportTestStart(sampleTest);

	yoloRT sample(initializeSampleParams(args));
	sample::gLogInfo << "Building and running a GPU inference engine for Onnx yolov5s" << std::endl;
	if (!sample.doMain())
	{
		std::cout << "all processing having problem" << std::endl;
	}

	return sample::gLogger.reportPass(sampleTest);

}