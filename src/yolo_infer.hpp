#pragma once
#include <vector>
#include <string>
#include <future>
#include "yi_type.hpp"


namespace yoloinfer
{
    using Boxes = std::vector<Box>;
    using Images = std::vector<Image>;

    struct YoloParam
    {
        std::string model_path;
        TaskType task_type;
        ModelType model_type;
        float score_thresh{0.4};
        float nms_thresh{0.45};
    };
    
    class YoloInfer
    {
    public:
        virtual ~YoloInfer() = default;
        virtual std::vector<Box> forward(const Image &image) = 0;
    protected:
        YoloInfer() = default;
    };

    std::shared_ptr<YoloInfer> CreateYoloInfer(const YoloParam& param);
}