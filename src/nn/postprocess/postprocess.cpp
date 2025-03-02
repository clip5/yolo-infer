#include "postprocess.hpp"
#include "decode.h"

namespace yoloinfer
{
    PostProcDet::PostProcDet(const Info &info) : PostProc(info)
    {
        decode_boxes_ = Tensor({info_.dims[0], max_boxes_ + 1, box_element_}, Tensor::Type::FP32);
    }

    std::vector<Box> PostProcDet::forward(const std::vector<Tensor> &tensors, const AffineMat &affine, void* stream)
    {
        if (tensors.size() != 1) {
            spdlog::get("logger")->error("PostProcDet only support one tensor");
            return {};
        }
        Tensor tensor = tensors[0];
        if(info_.model_type == ModelType::v5) {
            decode_detect(tensor, decode_boxes_, affine, info_.score_thr, info_.nms_thr, stream);
        } else if(info_.model_type == ModelType::v8) {
            decode_detectv8(tensor, decode_boxes_, affine, info_.score_thr, info_.nms_thr, stream);
        }
        decode_boxes_.syn(Device::gpu, Device::cpu);
        int count = std::min(max_boxes_, (uint32_t)((float*)decode_boxes_.data(Device::cpu))[0]);

        std::vector<Box> output;
        for (int i = 0; i < count; ++i)
        {
            float *pbox = (float*)decode_boxes_.data(Device::cpu) + 1 + i * box_element_;
            unsigned int label = pbox[5];
            int keepflag = pbox[6];

            auto rect = Box::Rect<float>{pbox[0], pbox[1], pbox[2]-pbox[0], pbox[3]-pbox[1], 0};
            if (keepflag == 1) {
                output.emplace_back(rect, pbox[4], label);
            }
        }

        return output;
    }


    PostProcObb::PostProcObb(const Info &info) : PostProc(info)
    {
        box_element_ = 8;
        decode_boxes_ = Tensor({info_.dims[0], max_boxes_ + 1, box_element_}, Tensor::Type::FP32);
    }

    std::vector<Box> PostProcObb::forward(const std::vector<Tensor> &tensors, const AffineMat &affine, void* stream)
    {
        if (tensors.size() != 1) {
            spdlog::get("logger")->error("PostProcObb only support one tensor");
            return {};
        }
        decode_detect_obb(tensors[0], decode_boxes_, affine, info_.score_thr, info_.nms_thr, stream);
        decode_boxes_.syn(Device::gpu, Device::cpu);
        int count = std::min(max_boxes_, (uint32_t)((float*)decode_boxes_.data(Device::cpu))[0]);

        std::vector<Box> output;
        for (int i = 0; i < count; ++i)
        {
            float *pbox = (float*)decode_boxes_.data(Device::cpu) + 1 + i * box_element_;
            unsigned int label = pbox[6];
            int keepflag = pbox[7];

            auto rect = Box::Rect<float>{pbox[0], pbox[1], pbox[2], pbox[3], pbox[4]};
            if (keepflag == 1) {
                output.emplace_back(rect, pbox[5], label);
            }
        }

        return output;
    }

    PostProcPose::PostProcPose(const Info &info) : PostProc(info)
    {
        box_element_ += info_.dims[2] - 5;
        decode_boxes_ = Tensor({info_.dims[0], max_boxes_ + 1, box_element_}, Tensor::Type::FP32);
        keypoint_num_ = (int)((info_.dims[2] - 5)/3);
    }

    std::vector<Box> PostProcPose::forward(const std::vector<Tensor> &tensors, const AffineMat &affine, void* stream)
    {
        if (tensors.size() != 1) {
            spdlog::get("logger")->error("PostProcPose only support one tensor");
            return {};
        }
        decode_pose(tensors[0], decode_boxes_, affine, info_.score_thr, info_.nms_thr, stream);
        decode_boxes_.syn(Device::gpu, Device::cpu);
        int count = std::min(max_boxes_, (uint32_t)((float*)decode_boxes_.data(Device::cpu))[0]);
        std::vector<Box> output;
        for (int i = 0; i < count; ++i)
        {
            float *pbox = (float*)decode_boxes_.data(Device::cpu) + 1 + i * box_element_;
            unsigned int label = pbox[5];
            int keepflag = pbox[6];

            auto rect = Box::Rect<float>{pbox[0], pbox[1], pbox[2] - pbox[0], pbox[3] - pbox[1], 0};
            if (keepflag == 1) {
                std::vector<Box::Point<float>> points;
                for(int j = 0; j < keypoint_num_; j++) {
                    points.push_back(Box::Point<float>{pbox[7+j*3], pbox[7+j*3+1]});
                }
                output.emplace_back(rect, points, pbox[4], label);
            }
        }

        return output;
    }

    PostProcSeg::PostProcSeg(const Info &info) : PostProc(info)
    {
        box_element_ = 8;
        decode_boxes_ = Tensor({info_.dims[0], max_boxes_ + 1, box_element_}, Tensor::Type::FP32);
    }


    void affine_project(float *matrix, float x, float y, float *ox, float *oy) 
    {
        *ox = matrix[0] * x + matrix[1] * y + matrix[2];
        *oy = matrix[3] * x + matrix[4] * y + matrix[5];
    }

    std::vector<Box> PostProcSeg::forward(const std::vector<Tensor> &tensors, const AffineMat &affine, void* stream)
    {
        decode_segv8(tensors[0], tensors[1], decode_boxes_, affine, info_.score_thr, info_.nms_thr, stream);
        decode_boxes_.syn(Device::gpu, Device::cpu);
        int count = std::min(max_boxes_, (uint32_t)((float*)decode_boxes_.data(Device::cpu))[0]);
        std::vector<Box> output;
        for (int i = 0; i < count; ++i)
        {
            float *pbox = (float*)decode_boxes_.data(Device::cpu) + 1 + i * box_element_;
            unsigned int label = pbox[5];
            int keepflag = pbox[6];
            int row_index = pbox[7];
            if (keepflag != 1) {
                continue;
            }
            auto rect = Box::Rect<float>{pbox[0], pbox[1], pbox[2] - pbox[0], pbox[3] - pbox[1], 0};

            float left, top, right, bottom;

            affine_project(affine.i2d(Device::cpu), pbox[0], pbox[1], &left, &top);
            affine_project(affine.i2d(Device::cpu), pbox[2], pbox[3], &right, &bottom);

            float box_width = right - left;
            float box_height = bottom - top;

            float scale_to_predict_x = tensors[1].shape()[3] / (float)info_.net_dims[3];
            float scale_to_predict_y = tensors[1].shape()[2] / (float)info_.net_dims[2];
            int mask_out_width = box_width * scale_to_predict_x + 0.5f;
            int mask_out_height = box_height * scale_to_predict_y + 0.5f;

            Box box;
            box.rect = rect;
            box.label = label;
            box.score = pbox[4];
            box.seg = std::make_shared<Image>(mask_out_width, mask_out_height);

            float *mask_weights = (float*)tensors[0].data(Device::gpu) + row_index * tensors[0].shape()[2]
                            + tensors[0].shape()[2] - tensors[1].shape()[1];
            decode_single_mask(left * scale_to_predict_x, top * scale_to_predict_y, mask_weights,
                                 (float*)tensors[1].data(Device::gpu),
                                 tensors[1].shape()[3], tensors[1].shape()[2], (unsigned char*)box.seg->data(Device::gpu),
                                 tensors[1].shape()[1], mask_out_width, mask_out_height, stream);
            box.seg->syn(Device::gpu, Device::cpu);
            output.emplace_back(box);
        }

        return output;
    }
}