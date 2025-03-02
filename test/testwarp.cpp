#include "affine.h"
#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>

using namespace yoloinfer;
int main(int argv, char** argc)
{
    cv::Mat src = cv::imread(argc[1]);
    
    Image imgSrc(src.cols, src.rows, ImageFmt::BGR);
    memcpy(imgSrc.data(Device::cpu), src.data, src.total() * src.elemSize());
    Image imgDst(640, 640, ImageFmt::BGR, DataType::FLOAT);

    Norm norm = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::RGB);
    AffineMat matrix;
    matrix.compute(std::make_tuple(src.cols, src.rows), std::make_tuple(640, 640));

    cudaStream_t stream{};
    cudaStreamCreate(&stream);
    imgSrc.syn(Device::cpu, Device::gpu);
    warp_affine_normal(imgSrc, imgDst, norm, matrix.d2i(Device::gpu), stream);
    imgDst.syn(Device::gpu, Device::cpu);
    cudaStreamDestroy(stream);

    cv::Mat dst = cv::Mat(imgDst.height(), imgDst.width(), CV_32FC1, imgDst.data(yoloinfer::Device::cpu));
    cv::imwrite(argc[2], dst*255);
    return 0;
}