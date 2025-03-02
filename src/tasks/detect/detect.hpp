#pragma once
#include <future>
#include "yolo_infer.hpp"
#include "database.hpp"
#include "postprocess.hpp"
#include "yolo_task.hpp"

namespace yoloinfer
{
    class Engine;
    class YoloDetect : public YoloInfer, public YoloTask
    {
        public:
            YoloDetect(const YoloParam& param);
            ~YoloDetect() = default;
            // forward
            std::vector<Box> forward(const Image &image);
            std::vector<Boxes> forward(const Images &images);
    };

}
