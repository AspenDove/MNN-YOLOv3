//  Created by Linzaer on 2019/11/15.
//  Copyright © 2019 Linzaer. All rights reserved.

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
    int w = 320, h = 240;

    UltraFace ultraface(mnn_path, w, h, 2, 0.65);
    pfpld PFLD(pfpld_mnn_path, 4);

    tbb::concurrent_bounded_queue<cv::Mat> qframe;
    tbb::concurrent_bounded_queue<std::pair<cv::Mat, std::vector<FaceInfo>>> qdetect;
    qframe.set_capacity(50);
    qdetect.set_capacity(5);

    cv::VideoCapture cap("/home/yealink/Desktop/mouth.mp4");
    cv::Mat frame;

    auto start = chrono::steady_clock::now();

    tbb::parallel_invoke(
        [&] {
        while(cap.isOpened())
        {
            if(cap.read(frame))
                qframe.push(frame.clone());
            else break;
        }
        cap.release();
        }, [&] {
        while(cap.isOpened())
        {
            cv::Mat frame;
            if(qframe.try_pop(frame))
            {
                std::vector<FaceInfo> face_info;
                ultraface.detect(frame, face_info);
                for (auto& face_box : face_info)
                {
                    PFLD.detect(frame, face_box);
                }
                qdetect.push({frame,face_info});
            }
        }
        cap.release(); 
        }, [&] {
        std::pair<cv::Mat,std::vector<FaceInfo>> detect;
        while(cap.isOpened())
        {
            if(qdetect.try_pop(detect))
            {
                for (FaceInfo &face : detect.second)
                {
                    cv::Point pt1(face.x1, face.y1); 
                    cv::Point pt2(face.x2, face.y2);
                    cv::rectangle(detect.first, pt1, pt2, cv::Scalar(0, 255, 0), 2);
                    for (auto coord : face.ls)
                        cv::circle(detect.first, cv::Point(coord.x, coord.y), 1, cv::Scalar(255,255,0), 1);
                }
                std::cout<<qdetect.size()<<std::endl;
                cv::namedWindow("show", CV_WINDOW_AUTOSIZE);
                cv::imshow("show", detect.first);
                cv::waitKey(33);
            }
        }
        while(qdetect.try_pop(detect));
        cap.release();
        cv::destroyAllWindows();
    });

    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "total time: " << elapsed.count() << "s" << endl;

    return 0;
}