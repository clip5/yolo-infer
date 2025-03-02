#include "yolo_infer.hpp"
#include "detect/detect.hpp"
#include "obb/obb.hpp"
#include "pose/pose.hpp"
#include "segment/segment.hpp"

namespace yoloinfer
{
    std::shared_ptr<YoloInfer> CreateYoloInfer(const YoloParam& param)
    {
        switch (param.task_type)
        {
        case TaskType::det:
            return std::make_shared<YoloDetect>(param);
        case TaskType::obb:
            return std::make_shared<YoloObb>(param);
        case TaskType::pose:
            return std::make_shared<YoloPose>(param);
        case TaskType::seg:
            return std::make_shared<YoloSeg>(param);
        default:
            return std::make_shared<YoloDetect>(param);
            break;
        }
    }

}