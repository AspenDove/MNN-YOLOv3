#include "pfpld.hpp"
#include "UltraFace.hpp"
using namespace std;

pfpld::pfpld(const std::string &mnn_path, int num_thread_)
{
    pfpld_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));

    MNN::ScheduleConfig config;
    config.numThread = num_thread_;
    MNN::BackendConfig backConfig;
    backConfig.precision = (MNN::BackendConfig::PrecisionMode)1;
    config.backendConfig = &backConfig;

    pfpld_session = pfpld_interpreter->createSession(config);
    input_tensor = pfpld_interpreter->getSessionInput(pfpld_session, nullptr);
}

pfpld::~pfpld()
{
    pfpld_interpreter->releaseModel();
    pfpld_interpreter->releaseSession(pfpld_session);
}

int pfpld::detect(const cv::Mat &raw_img, FaceInfo &face_box)
{
    int x1 = face_box.x1, x2 = face_box.x2, y1 = face_box.y1, y2 = face_box.y2;
    int w = x2 - x1 + 1;
    int h = y2 - y1 + 1;
    int size_w = max(w, h) * 0.9;
    int size_h = max(w, h) * 0.9;

    x1 += (w-size_w)/2;
    x2 = x1 + size_w;
    y1 += h / 2. - size_h * 0.4;
    y2 = y1 + size_h;

    int left = 0, top = 0, bottom = 0, right = 0;
    if (x1 < 0)
        left = -x1;
    if (y1 < 0)
        top = -y1;
    if (x2 >= raw_img.cols)
        right = x2 - raw_img.cols;
    if (y2 >= raw_img.rows)
        bottom = y2 - raw_img.rows;

    x1 = max(0, x1);
    y1 = max(0, y1);

    cv::Mat face_image;

    cv::copyMakeBorder(raw_img(cv::Rect(x1, y1, min(raw_img.cols, x2) - x1, min(raw_img.rows, y2) - y1)), face_image, top, bottom, left, right, cv::BORDER_CONSTANT, 0);

    cv::resize(face_image, face_image, cv::Size(112, 112));
    std::shared_ptr<MNN::CV::ImageProcess>(
        MNN::CV::ImageProcess::create(MNN::CV::RGB, MNN::CV::RGB, (const float[3]){0, 0, 0}, 3, (const float[3]){1 / 255.f, 1 / 255.f, 1 / 255.f}, 3))
        ->convert(face_image.data, 112, 112, face_image.step[0], input_tensor);

    auto start = chrono::steady_clock::now();
    // run network
    pfpld_interpreter->runSession(pfpld_session);

    // get output data
    string landms = "landms";
    string pose = "pose";
    MNN::Tensor *tensor_landms = pfpld_interpreter->getSessionOutput(pfpld_session, landms.c_str());
    MNN::Tensor *tensor_pose = pfpld_interpreter->getSessionOutput(pfpld_session, pose.c_str());
    // landmarks
    MNN::Tensor tensor_landms_host(tensor_landms, tensor_landms->getDimensionType());

    // count time consumption
    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    //cout << "inference time in pfpld: " << elapsed.count() << "s" << endl;

    for (int i = 0; i < 98; i++)
    {
        int x = tensor_landms->host<float>()[i * 2] * size_w;
        int y = tensor_landms->host<float>()[i * 2 + 1] * size_h;
        face_box.ls.push_back({x1 - left + x, y1 - bottom + y});
    }
    return 0;
}