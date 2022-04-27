#include "image.h"

using namespace base;

bool Image::draw( const std::vector<Detection>& dets, const std::string& wimgPath)
{
    for (Detection det : dets)
    {
        cv::rectangle(selforiImage, det.box, color[det.clsId], 3);
        std::string label = clsName[det.clsId] + ":" + std::to_string(round(det.conf, 2)).substr(0, 4);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, CV_FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::putText(selforiImage, label, cv::Point(det.box.x, det.box.y-10), CV_FONT_HERSHEY_SIMPLEX, 0.5, color[det.clsId], 2);
    }
    cv::imwrite(wimgPath, selforiImage);
    return true;
}

bool Image::writeImg(cv::Mat& image)
{
    
    cv::imwrite(imgPath, image);
    std::cout << "writeimg success" << std::endl;
    return false;
}

bool Image::wirteVideo()
{
    return false;
}

