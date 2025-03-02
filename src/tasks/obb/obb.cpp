#include "obb.hpp"
#include "engine.hpp"
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"

namespace yoloinfer
{

    YoloObb::YoloObb(const YoloParam &param) : YoloTask(param)
    {

        //1. load model
        Engine::EnginePara engine_para;
        engine_para.model_path = param_.model_path;
        engine_para.stream = stream_.get();
        engine_ = CreateEngine(engine_para); 

        //2. get input info and create preprocess
        auto input_dim = engine_->GetDim(0); 
        PreProc::Info preinfo;
        preinfo.width = input_dim[3];
        preinfo.height = input_dim[2];
        preinfo.model_type = param_.model_type;
        preproc_ = PreProc(preinfo);
        image_ = Image(preinfo.width, preinfo.height, ImageFmt::BGR, DataType::FLOAT);
        std::string dims_str = "Input dims: ";
        for (const auto& dim : input_dim) {
            dims_str += std::to_string(dim) + " ";
        }
        spdlog::info(dims_str);

        //3. get output info 
        auto output_dim = engine_->GetDim(1);
        PostProcObb::Info postinfo;
        postinfo.nms_thr = param_.nms_thresh;
        postinfo.score_thr = param_.score_thresh;
        postinfo.model_type = param_.model_type;
        postinfo.dims = output_dim;
        postinfo.affine = preproc_.matrix();
        postproc_ = std::make_shared<PostProcObb>(postinfo);
        std::string output_dims_str = "Output dims: ";
        for (const auto& dim : output_dim) {
            output_dims_str += std::to_string(dim) + " ";
        }
        spdlog::info(output_dims_str);
        //4. create forward buffer
        // buffer_out_ = Buffer(output_dim[1] * output_dim[2] * sizeof(float));
        buffer_out_ = Tensor(output_dim, Tensor::Type::FP32);
    }

    std::vector<Box> YoloObb::forward(const Image &image)
    {
        preproc_.process(image, image_, stream_.get());
        // stream_.sync();
        // image_.syn<Device::gpu, Device::cpu>();
        // cv::Mat preImg = cv::Mat(image_.height(), image_.width(), CV_32FC1, image_.data<Device::cpu>());
        // cv::imwrite("preImg.jpg", preImg*255);

        Tensor buffer_in;
        buffer_in = image_;
        engine_->Forward({buffer_in}, {buffer_out_}, stream_.get());
        std::vector<Box> output = postproc_->forward({buffer_out_}, preproc_.matrix(), stream_.get());
        return output;
    }

    std::vector<Boxes> YoloObb::forward(const Images &images)
    {
        std::vector<Boxes> outputs;
        for (const auto& image : images) {
            outputs.push_back(forward(image));
        }
        return outputs;
    }
}