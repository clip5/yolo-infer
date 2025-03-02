#include "preprocess.hpp"
#include "engine.hpp"
#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>

using namespace yoloinfer;
int main(int argv, char** argc)
{
    cudaStream_t stream{};
    cudaStreamCreate(&stream);
    
    const int width = 1280;
    const int height = 768;
    //0. read image
    cv::Mat src = cv::imread(argc[1]);
    
    Image imgSrc(src.cols, src.rows, ImageFmt::BGR);
    memcpy(imgSrc.data(yoloinfer::Device::cpu), src.data, src.total() * src.elemSize());
    Image imgDst(width, height, ImageFmt::BGR, DataType::FLOAT);

    //1. preprocess
    PreProc::Info preinfo;
    preinfo.width = width;
    preinfo.height = height;
    preinfo.model_type = ModelType::v8;

    PreProc preProc(preinfo);
    preProc.process(imgSrc, imgDst, stream);
    //2. forward
    Tensor bufferIn;
    bufferIn = imgDst;
    Engine::EnginePara enginePara;
    enginePara.model_path = "../data/models/yolov8n.engine";
    enginePara.stream = stream;
    std::shared_ptr<Engine> engine = CreateEngine(enginePara);
    std::vector<uint32_t> outDim = engine->GetDim(1);

    Tensor bufferOut({outDim[0],outDim[1],outDim[2]}, Tensor::Type::FP32);
    // for(auto i : outDim){
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;
    engine->Forward({bufferIn}, {bufferOut}, stream);
    
    
    cudaStreamDestroy(stream);

    // cv::Mat dst = cv::Mat(imgDst.height(), imgDst.width(), CV_32FC1, imgDst.data<Device::cpu>());
    // cv::imwrite(argc[2], dst*255);


    
    
    return 0;
}