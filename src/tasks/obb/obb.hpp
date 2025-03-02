#pragma once
#include <future>
#include "yolo_infer.hpp"
#include "yolo_task.hpp"
#include "database.hpp"
#include "postprocess.hpp"

namespace yoloinfer
{
    class Engine;
    class YoloObb : public YoloInfer, public YoloTask
    {
        public:
            YoloObb(const YoloParam& param);
            ~YoloObb() = default;
            // forward
            std::vector<Box> forward(const Image &image);
            std::vector<Boxes> forward(const Images &images);
    };

}
