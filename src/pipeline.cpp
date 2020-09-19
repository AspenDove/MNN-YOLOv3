#include "UltraFace.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
using namespace std;

#include "tbb/task_scheduler_init.h"
#include "tbb/concurrent_queue.h"
#include "tbb/tbb_thread.h"
#include "tbb/pipeline.h"
#include "tbb/parallel_invoke.h"
#include "tbb/concurrent_queue.h"
#include "pfpld.hpp"

int main(int argc, char **argv)
{
    string mnn_path = argv[1];
    string pfpld_mnn_path = argv[2];

    tbb::task_scheduler_init init(20); // Automatic number of threads

    int w = 640, h = 480;

    const int N = 10, M = 10;

    std::array<std::shared_ptr<UltraFace>, N> ultraface{
        std::make_shared<UltraFace>(mnn_path, w, h, 2, 0.65),
        std::make_shared<UltraFace>(mnn_path, w, h, 2, 0.65),
        std::make_shared<UltraFace>(mnn_path, w, h, 2, 0.65),
        std::make_shared<UltraFace>(mnn_path, w, h, 2, 0.65),
        std::make_shared<UltraFace>(mnn_path, w, h, 2, 0.65),
        std::make_shared<UltraFace>(mnn_path, w, h, 2, 0.65),
        std::make_shared<UltraFace>(mnn_path, w, h, 2, 0.65),
        std::make_shared<UltraFace>(mnn_path, w, h, 2, 0.65),
        std::make_shared<UltraFace>(mnn_path, w, h, 2, 0.65),
        std::make_shared<UltraFace>(mnn_path, w, h, 2, 0.65)};

    auto start = chrono::steady_clock::now();

    tbb::parallel_pipeline(
        10, tbb::make_filter<void, std::pair<std::shared_ptr<cv::Mat>, int>>(
                tbb::filter::serial_in_order, [&](tbb::flow_control &fc) -> std::pair<std::shared_ptr<cv::Mat>, int> {
                    static cv::VideoCapture cap("/home/yealink/Desktop/mouth.mp4");
                    static int frame = 0;
                    if (cap.isOpened())
                    {
                        std::shared_ptr<cv::Mat> pFrame(new cv::Mat);
                        if (!cap.read(*pFrame))
                        {
                            fc.stop();
                            return std::make_pair(nullptr, frame);
                        }
                        return std::make_pair(pFrame, frame++);
                    }
                    else
                    {
                        //cap.release();
                        fc.stop();
                        return std::make_pair(nullptr, frame);
                    }
                }) &
                tbb::make_filter<std::pair<std::shared_ptr<cv::Mat>, int>, std::pair<std::shared_ptr<cv::Mat>, int>>(tbb::filter::parallel, [&](std::pair<std::shared_ptr<cv::Mat>, int> pFrameid) {
                    std::shared_ptr<cv::Mat> pFrame;
                    int id;
                    std::tie(pFrame, id) = pFrameid;
                    cv::resize(*pFrame, *pFrame, cv::Size(w, h));

                    ultraface[id % N]->ultraface_interpreter->resizeTensor(ultraface[id % N]->input_tensor, {1, 3, h, w});
                    ultraface[id % N]->ultraface_interpreter->resizeSession(ultraface[id % N]->ultraface_session);
                    std::shared_ptr<MNN::CV::ImageProcess> pretreat(
                        MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, ultraface[id % N]->mean_vals, 3,
                                                      ultraface[id % N]->norm_vals, 3));
                    pretreat->convert(pFrame->data, w, h, pFrame->step[0], ultraface[id % N]->input_tensor);
                    return std::make_pair(pFrame, id);
                }) &
                tbb::make_filter<std::pair<std::shared_ptr<cv::Mat>, int>, std::pair<std::shared_ptr<cv::Mat>, int>>(tbb::filter::parallel, [&](std::pair<std::shared_ptr<cv::Mat>, int> pFrameid) -> std::pair<std::shared_ptr<cv::Mat>, int> {
                    int id = pFrameid.second;
                    auto start = chrono::steady_clock::now();
                    // run network
                    ultraface[id % N]->ultraface_interpreter->runSession(ultraface[id % N]->ultraface_session);

                    auto end = chrono::steady_clock::now();
                    chrono::duration<double> elapsed = end - start;
                    cout << "inference time:" << elapsed.count() << " s" << endl;

                    return pFrameid;
                }) &
                tbb::make_filter<std::pair<std::shared_ptr<cv::Mat>, int>, std::pair<std::shared_ptr<std::vector<FaceInfo>>, std::shared_ptr<cv::Mat>>>(tbb::filter::parallel, [&](std::pair<std::shared_ptr<cv::Mat>, int> pFrameid) {
                    std::vector<FaceInfo> face_list;
                    std::shared_ptr<cv::Mat> pFrame;
                    int id;
                    std::tie(pFrame, id) = pFrameid;
                    string scores = "scores";
                    string boxes = "boxes";
                    // get output data
                    MNN::Tensor *tensor_scores = ultraface[id % N]->ultraface_interpreter->getSessionOutput(ultraface[id % N]->ultraface_session, scores.c_str());
                    MNN::Tensor *tensor_boxes = ultraface[id % N]->ultraface_interpreter->getSessionOutput(ultraface[id % N]->ultraface_session, boxes.c_str());

                    std::vector<FaceInfo> bbox_collection;

                    ultraface[id % N]->generateBBox(bbox_collection, tensor_scores, tensor_boxes, cv::Size(pFrame->cols, pFrame->rows));
                    ultraface[id % N]->nms(bbox_collection, face_list);
                    return std::make_pair(std::make_shared<std::vector<FaceInfo>>(face_list), pFrame);
                }) &
                tbb::make_filter<std::pair<std::shared_ptr<std::vector<FaceInfo>>, std::shared_ptr<cv::Mat>>, std::pair<std::shared_ptr<cv::Mat>, std::shared_ptr<std::vector<FaceInfo>>>>(tbb::filter::parallel, [&](std::pair<std::shared_ptr<std::vector<FaceInfo>>, std::shared_ptr<cv::Mat>> result) {
                    std::shared_ptr<std::vector<FaceInfo>> pDetection;
                    std::shared_ptr<cv::Mat> pFrame;
                    std::tie(pDetection, pFrame) = result;

                    cv::Mat face_image;
                    pfpld PFLD(pfpld_mnn_path, 4);
                    for (FaceInfo &face : *pDetection)
                    {
                        PFLD.detect(*pFrame, face);
                    }
                    return std::make_pair(pFrame, pDetection);
                }) &
                tbb::make_filter<std::pair<std::shared_ptr<cv::Mat>, std::shared_ptr<std::vector<FaceInfo>>>, void>(tbb::filter::serial_in_order, [&](std::pair<std::shared_ptr<cv::Mat>, std::shared_ptr<std::vector<FaceInfo>>> result) {
                    std::shared_ptr<std::vector<FaceInfo>> pDetection;
                    std::shared_ptr<cv::Mat> pFrame;
                    std::tie(pFrame, pDetection) = result;
                    //show detection results
                    for (FaceInfo &face : *pDetection)
                    {
                        cv::Point pt1(face.x1, face.y1);
                        cv::Point pt2(face.x2, face.y2);
                        cv::rectangle(*pFrame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
                        for (auto coord : face.ls)
                            cv::circle(*pFrame, cv::Point(coord.x, coord.y), 1, cv::Scalar(255, 255, 0), 1);
                    }
                    cv::namedWindow("show", CV_WINDOW_AUTOSIZE);
                    cv::imshow("show", *pFrame);
                    cv::waitKey(1);
                }));
    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "total time: " << elapsed.count() << "s" << endl;

    double sum_cost = 0.0;
    int num_pics = 0;

    return 0;
}