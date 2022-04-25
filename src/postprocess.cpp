#include "postprocess.h"

bool PostProcess::getMaxclsConf(std::vector<float>::iterator it, const int& numCls, float& maxclsConf, int& bestclsId)
{
    maxclsConf = 0;
    bestclsId = 5;
    for (int i = 5; i < numCls + 5; i++)
    {
        if (it[i] > maxclsConf)
        {
            maxclsConf = it[i];
            bestclsId = i - 5;
        }
    }

    return true;
}

bool PostProcess::scaleBox2OriSize(cv::Rect& box)
{
    float gain = std::min((float)selfinputFrameSize.width / (float)selforiginalFrameSize.width, (float)selfinputFrameSize.height / (float)selforiginalFrameSize.height);
    int pad[2] = { (int)(((float)selfinputFrameSize.width - (float)selforiginalFrameSize.width * gain) / 2.0f),
                (int)(((float)selfinputFrameSize.height - (float)selforiginalFrameSize.height * gain) / 2.0f) };

    box.x = (int)std::round((box.x - pad[0]) / gain);
    box.y = (int)std::round((box.y - pad[1]) / gain);
    box.width = (int)std::round(box.width / gain);
    box.height = (int)std::round(box.height / gain);

    return true;
}


bool PostProcess::outputs2dets(float* prob, const nvinfer1::Dims&  outputdims)
{
    size_t nc = outputdims.d[2] - 5;
    size_t count = outputdims.d[1] * outputdims.d[2];
    std::vector<float> output(prob, prob + count);
    std::vector<cv::Rect> boxs;
    std::vector<float> clsIds;
    std::vector<float> confs;
    //找出大于目标置信度阈值的框，与其框中的目标类别。
    for (auto it=output.begin(); it != output.end(); it+=outputdims.d[2])
    {
        float objConf = it[4];
        if (objConf > selfconf)
        {
            int centerX = it[0];
            int centerY = it[1];
            int width = it[2];
            int height = it[3];
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float maxclsConf;
            int bestclasId;

            getMaxclsConf(it, nc, maxclsConf, bestclasId);
            // co_conf  is class and object ;
            float co_conf = maxclsConf * objConf;

            confs.emplace_back(co_conf);
            clsIds.emplace_back(bestclasId);
            boxs.emplace_back(left, top, width, height);

        }
    }
    //通过NMS方法找到最优框
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxs, confs, selfconf, selfiou, indices);
    for (int idx:indices)
    {
        Detection det;
        det.box = boxs[idx];
        det.clsId = clsIds[idx];
        det.conf = confs[idx];     
        scaleBox2OriSize(det.box);
        detects.emplace_back(det);
    }
    return true;
}
