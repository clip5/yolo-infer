#pragma once
#include "common.hpp"
#include "database.hpp"
#include "affine.h"

namespace yoloinfer 
{

class Tensor;
class AffineMat
{
    public:
    AffineMat();
    ~AffineMat() = default;

    float* i2d(const Device& d) const;
    float* d2i(const Device& d) const;

    void compute(const std::tuple<int, int>& from, const std::tuple<int, int>& to);

    private:
        std::shared_ptr<Tensor> i2d_;
        std::shared_ptr<Tensor> d2i_;
        uint32_t size_{6};
};


class PreProc
{
    public:
        struct Info
        {
            unsigned int width;
            unsigned int height;
            ModelType model_type;
        };
        
    public:
        PreProc() = default;
        ~PreProc() = default;
        PreProc(Info info);
        const Info& info() const { return info_; };
        void process(const Image& src, Image& dst, void* stream = nullptr);
        AffineMat& matrix() { return matrix_; };
    private:
        Info info_;
        Norm norm_;
        AffineMat matrix_;
        // AffineMat matrix_;
};
}