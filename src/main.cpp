#include "do_inference.h"
#include "logging.h"
#include "parser.h"
#include "onnx2trt.h"


int main(int argc, char** argv)
{

	Parser* par = new Parser();
	std::cout << "init par success" << std::endl;
	SampleParams zparams;
	if (!par->paraseArgv(argc, argv, zparams))
	{
		
		return false;
	}

	const std::string gSampleName = "TensorRT.sample_yolov5s";
	auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

	sample::gLogger.reportTestStart(sampleTest);

	//直接通过tensorrt进行推理
	yoloRT sample(zparams);

	//通过onnx模型转换为tensorrt  再推理
	//Onnx2Trt sample(zparams);
	sample::gLogInfo << "Building and running a GPU inference engine for  yolov5s" << std::endl;
	if (!sample.doMain())
	{
		std::cout << "all processing having problem" << std::endl;
		return false;
	}

	return sample::gLogger.reportPass(sampleTest);

}