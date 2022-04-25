#include "preprocess.h"

void Preprocess::letterbox(const cv::Mat& image, cv::Mat& outImage, const cv::Size& newShape = cv::Size(640, 640), const cv::Scalar& color = cv::Scalar(114, 114, 114), bool auto_ = false, bool scaleFill = false, bool scaleUp = true, int stride = 32)
{
    cv::Size shape = image.size();

    float r = std::min((float)newShape.height / (float)shape.height,
        (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int newUnpad[2]{ (int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] || shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }
    else {
        outImage = image;
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

cv::Mat Preprocess::preAll(const cv::Mat& image, const cv::Size& image_size)
{
    cv::Mat media;
    letterbox(image, media);
    cv::imwrite("./media.jpg", media);
    std::cout << "media size :" << media.size << std::endl;
    cv::Mat input_data;
    input_data = cv::dnn::blobFromImage(media, 1.0 / 255.0, image_size, cv::Scalar(0, 0, 0), true, false, CV_32F);
    return input_data;
}


cv::Mat Preprocess::preprocess_img(cv::Mat& img)
{
    int w, h, x, y;
    float r_w = 640 / (img.cols * 1.0);
    float r_h = 640 / (img.rows * 1.0);
    if (r_h > r_w) {
        w = 640;
        h = r_w * img.rows;
        x = 0;
        y = (640 - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = 640;
        x = (640 - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    //cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    //auto start = std::chrono::system_clock::now();
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    //resizeByNN(img.data, re.data, img.rows, img.cols, img.channels(), re.rows, re.cols);
    //auto end = std::chrono::system_clock::now();
    //std::cout<< "img resize: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    cv::Mat out(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    cv::imwrite("./media.jpg", out);
    return out;
}