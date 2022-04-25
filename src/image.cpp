#include "image.h"
#include <iomanip>
#include <sstream>

//Image::Image(const std::string imgpath) 
//{
//    std::cout << "InitImage Object Sucess" << std::endl;
//    srand(time(0));
//    for (int i = 0; i < 80; i++) {
//        int b = rand() % 256;
//        int g = rand() % 256;
//        int r = rand() % 256;
//        color.push_back(cv::Scalar(b, g, r));
//    }
//}
float Image::round(float src, int bits)
{
    //std::cout.precision(2);
    std::stringstream ss;
    float f;
    ss << std::fixed << std::setprecision(bits) << src;
    ss >> f;

    return f;

}


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

