#ifndef PROCESSING_HEADER
#define PROCESSING_HEADER

#include <opencv2/opencv.hpp>
#include <vector>

const int OUTPUT_SIZE = 388;
const int INPUT_SIZE = 572;

int stride_size(int img_len, int crop_num, int crop_size)
{
    return (img_len - crop_size) / (crop_num - 1);
}

cv::Mat cropping(cv::Mat img, int crop_size, int dimx, int dimy)
{

    cv::Rect rect(dimx, dimy, crop_size, crop_size);

    return img(rect);
}

std::vector<cv::Mat> multi_cropping(cv::Mat image, int crop_size, int crop_num, int crop_num2)
{
    auto width = image.cols;
    auto heigth = image.rows;

    std::vector<cv::Mat> croppedImages;

    auto dimy_stride = stride_size(heigth, crop_num, crop_size);
    auto dimx_stride = stride_size(width, crop_num2, crop_size);

    for (int i = 0; i < crop_num; i++)
        for (int j = 0; j < crop_num2; j++)
            croppedImages.push_back(cropping(image, crop_size, dimx_stride * j, dimy_stride * i));

    return croppedImages;
}

cv::Mat normalization2(cv::Mat image, int max, int min)
{
    double imgMax, imgMin;
    cv::minMaxLoc(image, &imgMin, &imgMax);
    // std::cout << imgMax << "    -----     " << imgMin << std::endl;
    auto ret = (image - imgMin) * (float)(max - min) / (imgMax - imgMin) + min;
    return ret;
}

std::vector<cv::Mat> preProcessing(cv::Mat image)
{
    int crop_size = INPUT_SIZE;
    int pad_size = int((INPUT_SIZE - OUTPUT_SIZE) / 2);

    cv::Mat reImage;
    cv::cvtColor(image, reImage, cv::COLOR_BGR2GRAY);

    cv::Mat padded;
    cv::copyMakeBorder(reImage, padded, pad_size, pad_size, pad_size, pad_size, cv::BORDER_REFLECT);

    // cv::imwrite("padded.png", padded);

    auto croppedImages = multi_cropping(padded, crop_size, 2, 2);

    std::vector<cv::Mat> processedImg;

    //maybe cropped image size is 4
    int i = 0;
    for (auto cimg : croppedImages)
    {
        // cv::imwrite("t" + std::to_string(i) + ".png", cimg);
        cv::Mat fMat;
        cimg.convertTo(fMat, CV_32F);

        // std::cout << "before nomalize : " << fMat.at<float>(0, 0) << std::endl;
        auto ret = normalization2(fMat, 1, 0);
        // std::cout << "after nomalize: " << ret.at<float>(0, 0) << std::endl;
        processedImg.push_back(ret.clone());
        i++;
    }

    return processedImg;
}

cv::Mat division_array(int crop_size, int xcrop, int ycrop, int dim1, int dim2)
{
    cv::Mat zMat(cv::Size(dim2, dim1), CV_32F);
    zMat.setTo(cv::Scalar::all(0));

    auto dim1_stride = stride_size(dim1, ycrop, crop_size); //Y
    auto dim2_stride = stride_size(dim2, xcrop, crop_size); //X

    for (int i = 0; i < ycrop; ++i)
    {
        for (int j = 0; j < xcrop; ++j)
        {
            int tempIdx = 0;
            for (int row = i * dim1_stride; row < i * dim1_stride + crop_size; ++row)
            {
                for (int col = j * dim2_stride; col < j * dim2_stride + crop_size; ++col)
                {
                    float &t = zMat.at<float>(row, col);
                    auto value = 1;
                    t += value;
                }
            }
        }
    }

    return zMat;
}

cv::Mat image_concatenate(const std::vector<cv::Mat> &images, int ycrop, int xcrop, int dim1, int dim2)
{

    cv::Mat ret(cv::Size(dim2, dim1), CV_32FC1);
    int crop_size = images[0].rows;
    auto dim1_stride = stride_size(dim1, ycrop, crop_size); //Y
    auto dim2_stride = stride_size(dim2, xcrop, crop_size); //X

    int index = 0;
    for (int i = 0; i < ycrop; ++i)
    {
        for (int j = 0; j < xcrop; ++j)
        {
            int tempIdx = 0;
            int rowTemp = 0;
            int colTemp = 0;
            for (int row = i * dim1_stride; row < i * dim1_stride + crop_size; ++row)
            {
                colTemp = 0;
                for (int col = j * dim2_stride; col < j * dim2_stride + crop_size; ++col)
                {
                    float &t = ret.at<float>(row, col);
                    auto value = images[index].at<float>(rowTemp, colTemp++);
                    t += value;
                }

                rowTemp++;
            }

            index++;
        }
    }

    return ret;
}

cv::Mat polarize(cv::Mat mat)
{
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            auto &value = mat.at<float>(i, j);
            if (value >= 0.5)
                value = 1;
            else if (value < 0.5)
                value = 0;
        }
    }

    return mat;
}

cv::Mat postProcessing(float **temp)
{
    auto div_arr = division_array(388, 2, 2, 512, 512);

    int maxIndex = -1;

    std::vector<cv::Mat> vecTest;
    float a[388 * 388];
    float b[388 * 388];
    float c[388 * 388];

    std::ofstream outfile("output.txt");
    for (int i = 0; i < 4; ++i)
    {

        memcpy(a, &temp[i][0], 388 * 388 * sizeof(float));
        memcpy(b, &temp[i][388 * 388], 388 * 388 * sizeof(float));

        cv::Mat ct(cv::Size(388, 388), CV_32FC1);

        for (int i = 0; i < 388 * 388; i++)
        {
            if (a[i] > b[i])
                c[i] = 0;
            else
                c[i] = 1;
        }

        memcpy(ct.data, c, 388 * 388 * sizeof(float));

        for (int k = 0; k < 388 * 388; k++)
            outfile << a[k] << " ";
        outfile << "\n";
        for (int k = 0; k < 388 * 388; k++)
            outfile << b[k] << " ";
        outfile << "\n";
        for (int k = 0; k < 388 * 388; k++)
            outfile << c[k] << " ";
        outfile << "\n";
        outfile << "======================================================================================================\n";

        vecTest.push_back(ct.clone());
    }

    auto img_cont = image_concatenate(vecTest, 2, 2, 512, 512);
    img_cont = img_cont / div_arr;
    auto ret = polarize(img_cont) * 255;
    return ret;
}

#endif