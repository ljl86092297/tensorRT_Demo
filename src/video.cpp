#include "video.h"

using namespace base;


Video::Video(const std::string& videoUrl)
{
	cap = cv::VideoCapture(videoUrl);
	std::cout << "InitFrame Object Sucess" << std::endl;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(cv::Scalar(b, g, r));
	}
}

bool Video::get_frame(cv::Mat& frame)
{
	if (cap.isOpened())
	{
		cap >> frame;
		if (frame.size[0] == 0 or frame.size[1] == 0)
		{
			return false;
		}
		return true;
	}
	return false;
}

bool Video::create_videoMemory(const std::string& saveVideoPath, const int& fps)
{
	if (cap.get(3) > 0 and cap.get(4))
	{
		saveSize = cv::Size(cap.get(3), cap.get(4));
		//saveSize = cv::Size(1280, 720);
		std::cout << saveSize << std::endl;
	}
	else {
		std::cout << "can't get your capture height or width ,please update code in <video.cpp> that have function <create_videoMemory>" << std::endl;
	}
	//cv::Size(height, width) 请输入你视频或者摄像头的宽高并替换saveSize;
	//CV_FOURCC('D', 'I', 'V', '3')
	//CV_FOURCC('M', 'J', 'P', 'G')
	output.open(saveVideoPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, saveSize);
	return true;
}

bool Video::write_video(const cv::Mat& frame)
{
	//std::cout << frame.size << std::endl;
	output << frame;
	return false;
}

bool Video::release()
{
	output.release();
	return false;
}


bool Video::draw(cv::Mat& frame, const std::vector<Detection>& dets)
{
    for (Detection det : dets)
    {
        cv::rectangle(frame, det.box, color[det.clsId], 3);
        std::string label = clsName[det.clsId] + ":" + std::to_string(round(det.conf, 2)).substr(0, 4);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, CV_FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::putText(frame, label, cv::Point(det.box.x, det.box.y-10), CV_FONT_HERSHEY_SIMPLEX, 0.5, color[det.clsId], 2);
    }
    return true;
}