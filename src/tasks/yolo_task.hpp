#pragma once
#include <future>
#include "yolo_infer.hpp"
#include "database.hpp"
#include "preprocess.hpp"

namespace yoloinfer
{
    class Engine;
    class PostProc;
    class YoloTask 
    {
        public:
            YoloTask(const YoloParam& param) : param_(param) {};
            ~YoloTask() = default;
        protected:
            YoloParam param_;
            Stream stream_;
            ModelType model_type_;

            Image image_;
            Tensor buffer_out_;
            
            PreProc preproc_;
            std::shared_ptr<PostProc> postproc_;
            std::shared_ptr<Engine> engine_;
    };
}