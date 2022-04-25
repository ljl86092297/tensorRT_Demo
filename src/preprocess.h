#pragma once
#include <opencv2/opencv.hpp>

class Preprocess
{

public:

    cv::Mat preAll(const cv::Mat& image, const cv::Size& image_size);
    cv::Mat preprocess_img(cv::Mat& img);
private:
    void letterbox(const cv::Mat& image, cv::Mat& outImage,
        const cv::Size& newShape, const cv::Scalar& color, bool auto_,
        bool scaleFill, bool scaleUp, int stride);
    

};