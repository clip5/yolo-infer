#include "preprocess.hpp"
#include "affine.h"
#include "opencv2/opencv.hpp"
    
namespace yoloinfer 
{
    AffineMat::AffineMat() : size_(6)
    {
        i2d_ = std::make_shared<Tensor>(std::vector<uint32_t>({size_}), Tensor::Type::FP32);
        d2i_ = std::make_shared<Tensor>(std::vector<uint32_t>({size_}), Tensor::Type::FP32);
    }

    void AffineMat::compute(const std::tuple<int, int>& from, const std::tuple<int, int>& to)
    {
        float* i2d = (float*)i2d_->data(Device::cpu);
        float* d2i = (float*)d2i_->data(Device::cpu);
        
        float scale_x = std::get<0>(to) / (float)std::get<0>(from);
        float scale_y = std::get<1>(to) / (float)std::get<1>(from);
        float scale = std::min(scale_x, scale_y);
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * std::get<0>(from) * 0.5 + std::get<0>(to) * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * std::get<1>(from) * 0.5 + std::get<1>(to) * 0.5 + scale * 0.5 - 0.5;

        double D = i2d[0] * i2d[4] - i2d[1] * i2d[3];
        D = D != 0. ? double(1.) / D : double(0.);
        double A11 = i2d[4] * D, A22 = i2d[0] * D, A12 = -i2d[1] * D, A21 = -i2d[3] * D;
        double b1 = -A11 * i2d[2] - A12 * i2d[5];
        double b2 = -A21 * i2d[2] - A22 * i2d[5];

        d2i[0] = A11;
        d2i[1] = A12;
        d2i[2] = b1;
        d2i[3] = A21;
        d2i[4] = A22;
        d2i[5] = b2;
        
        i2d_->syn(Device::cpu, Device::gpu);
        d2i_->syn(Device::cpu, Device::gpu);
    }

    float* AffineMat::d2i(const Device& device) const
    {
        return (float*)d2i_->data(device);
    }

    float* AffineMat::i2d(const Device& device) const
    {
        return (float*)i2d_->data(device);
    }

    PreProc::PreProc(Info info) : info_(info)
    {
        const auto& model_type = info_.model_type;
        if (model_type == ModelType::v5 || model_type == ModelType::v6 || model_type == ModelType::v7
        || model_type == ModelType::v8 || model_type == ModelType::v9 || model_type == ModelType::v10) {
            norm_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::RGB);
        } else if (model_type == ModelType::x) {
            norm_ = Norm::None();
        }
    }         
    void PreProc::process(const Image& src, Image& dst, void* stream)
    {
        matrix_.compute(std::make_tuple(src.width(), src.height()), std::make_tuple(info_.width, info_.height));
        src.syn(Device::cpu, Device::gpu);
        warp_affine_normal(src, dst, norm_, matrix_.d2i(Device::gpu), stream);
    }
    
}