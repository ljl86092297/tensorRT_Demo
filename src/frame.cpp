#include "frame.h"


//Frame::Frame(const std::string imgpath) 
//{
//    std::cout << "InitFrame Object Sucess" << std::endl;
//    srand(time(0));
//    for (int i = 0; i < 80; i++) {
//        int b = rand() % 256;
//        int g = rand() % 256;
//        int r = rand() % 256;
//        color.push_back(cv::Scalar(b, g, r));
//    }
//}

bool Frame::draw(cv::Mat& frame, const std::vector<Detection>& dets)
{
    for (Detection det : dets)
    {
        cv::rectangle(frame, det.box, cv::Scalar(0, 0, 255), 3);
        std::string label = clsName[det.clsId] + ":" + std::to_string(det.conf);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, CV_FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::putText(frame, label, cv::Point(det.box.x, det.box.y), CV_FONT_HERSHEY_SIMPLEX, 1, color[det.clsId], 2);
    }
    cv::imwrite(imgPath, frame);
    return true;
}

bool Frame::writeImg(cv::Mat& frame)
{
    
    cv::imwrite(imgPath, frame);
    std::cout << "writeimg success" << std::endl;
    return false;
}

bool Frame::wirteVideo()
{
    return false;
}

