#include <iostream>
#include <string>
#include <memory>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "processing.h"

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char *msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }

    Severity reportableSeverity;
};

Logger gLogger;

int INPUT_WIDTH = 572;
int INPUT_HEIGHT = 572;

void preProcess(const cv::Mat &img, std::vector<cv::Mat> &input_channels)
{
    cv::Mat sample;
    int num_channels_ = 3;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    cv::Size input_geometry_(INPUT_WIDTH, INPUT_HEIGHT);
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_, 0.0, 0.0, cv::INTER_CUBIC);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized = sample_float / 255.0;
    // cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
    * input layer of the network because it is wrapped by the cv::Mat
    * objects in input_channels. */
    cv::split(sample_normalized, input_channels);
}

void writeMatToFile(const cv::Mat &m, const char *filename)
{
    using namespace std;
    ofstream fout(filename);

    if (!fout)
    {
        cout << "File Not Opened" << endl;
        return;
    }

    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            fout << m.at<float>(i, j) << "\t";
        }
        fout << endl;
    }

    fout.close();
}

int main(int argc, char *argv[])
{
    using namespace nvinfer1;
    Logger gLogger;
    auto runtime = nvinfer1::createInferRuntime(gLogger);
    std::string model = "unet.trt";
    auto file = fopen(model.c_str(), "r");

    fseek(file, 0, SEEK_END);
    size_t fsz = ftell(file);
    auto tbuffer = std::make_unique<char[]>(fsz);

    fseek(file, 0, SEEK_SET);
    size_t rsz = fread(tbuffer.get(), sizeof(char), fsz, file);
    auto engine = runtime->deserializeCudaEngine(tbuffer.get(), rsz, nullptr);

    if (!engine)
    {
        std::cout << "Cannot reading trt file" << std::endl;
        return -1;
    }
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto context = engine->createExecutionContext();

    // std::cout << "NB bindings :" << engine->getNbBindings() << std::endl;
    // std::cout << "input binding dimensions " << engine->getBindingDimensions(0).nbDims << std::endl;
    // std::cout << "input binding d 0 " << engine->getBindingDimensions(0).d[0] << std::endl;
    // std::cout << "input binding d 1 " << engine->getBindingDimensions(0).d[1] << std::endl;
    // std::cout << "input binding d 2 " << engine->getBindingDimensions(0).d[2] << std::endl;

    // std::cout << "Ouput binding dimensions " << engine->getBindingDimensions(1).nbDims << std::endl;
    // std::cout << "Ouput binding d 0 " << engine->getBindingDimensions(1).d[0] << std::endl;
    // std::cout << "Ouput binding d 1 " << engine->getBindingDimensions(1).d[1] << std::endl;
    // std::cout << "Ouput binding d 2 " << engine->getBindingDimensions(1).d[2] << std::endl;
    // auto indexoutput = engine->getBindingIndex("output");
    // std::cout << "Bind index " << indexoutput << std::endl;

    void *buffer[2];
    //todo change
    int batchSize = 1;
    int outsize = 388 * 388 * 2;
    int insize = 572 * 572;
    cudaMalloc(&buffer[0], batchSize * insize * sizeof(float));
    cudaMalloc(&buffer[1], batchSize * outsize * sizeof(float));

    cv::Mat img;
    img = cv::imread("27.png");
    if (img.empty())
    {
        std::cout << "img error" << std::endl;
        return -1;
    }

    // std::cout << img.rows << " " << img.cols << std::endl;
    // std::vector<cv::Mat> imgs;
    // preProcess(img, imgs);
    // std::vector<float> inputtest(572 * 572);

    // for (size_t i = 0; i < 3; ++i)
    // {
    //     float *data = imgs[i].ptr<float>(0, 0);
    //     const std::size_t offset = 572 * 572 * i;
    //     std::copy(data, data + (572 * 572), inputtest.begin() + offset);
    // }

    std::cout << "Start inference" << std::endl;
    auto processedImgs = preProcessing(img);
    if (processedImgs.size() != 4)
    {
        std::cout << "Preprocessing error!" << std::endl;
        return -1;
    }
    float **test = new float *[4];
    for (int i = 0; i < 4; i++)
    {
        test[i] = new float[388 * 388 * 2];
    }

    int index = 0;
    for (const auto &img : processedImgs)
    {
        // std::cout << index << " : " << img.type() << std::endl;
        // cv::Mat fMat;
        // img.convertTo(fMat, CV_32FC1);
        // cv::imwrite("t" + std::to_string(index) + ".jpg", fMat);
        std::string fname = std::to_string(index) + "_.txt";
        writeMatToFile(img, fname.c_str());
        cudaMemcpyAsync(buffer[0], img.data, 572 * 572 * sizeof(float), cudaMemcpyHostToDevice, stream);

        context->enqueue(1, &buffer[0], stream, nullptr);
        cudaStreamSynchronize(stream);

        cudaMemcpyAsync(&test[index++][0], buffer[1], 2 * 388 * 388 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

    auto retImg = postProcessing(test);

    cv::imwrite("finanl.jpg", retImg);

    // for (int i =0 ; i< 500 ; i++)
    //     std::cout << test[i];

    cudaFree(buffer[0]);
    cudaFree(buffer[1]);
    std::cout << "finish inference" << std::endl;
    engine->destroy();

    for (int i = 0; i < 4; i++)
        delete test[i];

    delete[] test;
    // runtime->destroy();
    return 0;
}
