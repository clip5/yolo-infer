#pragma once

#include "database.hpp"
#include "common.hpp"
#include "preprocess.hpp"

namespace yoloinfer
{

class PostProc
{
    public:
    struct Info
    {
        float nms_thr;
        float score_thr;
        ModelType model_type;
        std::vector<uint32_t> dims;
        std::vector<uint32_t> net_dims;
        AffineMat affine;
    };
    public:
        PostProc() = default;
        PostProc(const Info &info) : info_(info) {};
        virtual std::vector<Box> forward(const std::vector<Tensor> &tensors, const AffineMat &affine, void* stream=nullptr) = 0;

    protected:
        Info info_;
        Tensor decode_boxes_;
        uint32_t max_boxes_ = 1024;
        uint32_t box_element_ = 7;
};


class PostProcDet : public PostProc
{
    public:
        PostProcDet() = default;
        PostProcDet(const Info &info);
        std::vector<Box> forward(const std::vector<Tensor> &tensors, const AffineMat &affine, void* stream=nullptr);

};


class PostProcObb : public PostProc
{
    public:
        PostProcObb() = default;
        PostProcObb(const Info &info);
        std::vector<Box> forward(const std::vector<Tensor> &tensors, const AffineMat &affine, void* stream=nullptr);

};


class PostProcPose : public PostProc
{
    public:
        PostProcPose() = default;
        PostProcPose(const Info &info);
        std::vector<Box> forward(const std::vector<Tensor> &tensors, const AffineMat &affine, void* stream=nullptr);

    private:
        uint32_t keypoint_num_ = 17;
};

class PostProcSeg : public PostProc
{
    public:
        PostProcSeg() = default;
        PostProcSeg(const Info &info);
        std::vector<Box> forward(const std::vector<Tensor> &tensors, const AffineMat &affine, void* stream=nullptr);
};

}
