#pragma once
#include <future>
#include "yolo_infer.hpp"
#include "yolo_task.hpp"
#include "database.hpp"
#include "postprocess.hpp"

namespace yoloinfer
{
    class Engine;
    class YoloPose : public YoloInfer, public YoloTask
    {
        public:
            YoloPose(const YoloParam& param);
            ~YoloPose() = default;
            // forward
            std::vector<Box> forward(const Image &image);
            std::vector<Boxes> forward(const Images &images);
    };
}
