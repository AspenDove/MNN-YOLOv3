
#ifndef pfpld_hpp
#define pfpld_hpp

#include "Interpreter.hpp"
#include "UltraFace.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include "ImageProcess.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>


class pfpld
{
public:
    pfpld(const std::string &mnn_path, int num_thread_ =4);

    ~pfpld();

    int detect(const cv::Mat &raw_img, FaceInfo &face_box);

private:
    std::shared_ptr<MNN::Interpreter> pfpld_interpreter;
    MNN::Session *pfpld_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;
};

#endif