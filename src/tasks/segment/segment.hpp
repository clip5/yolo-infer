#pragma once
#include <future>
#include "yolo_infer.hpp"
#include "yolo_task.hpp"
#include "database.hpp"
#include "postprocess.hpp"

namespace yoloinfer
{
    class Engine;
    class YoloSeg : public YoloInfer, public YoloTask
    {
        public:
            YoloSeg(const YoloParam& param);
            ~YoloSeg() = default;
            // forward
            std::vector<Box> forward(const Image &image);
            std::vector<Boxes> forward(const Images &images);
        private:
            Tensor buffer_out_seg_;
    };
}
