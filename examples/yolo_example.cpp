#include "yolo_infer.hpp"
#include "opencv2/opencv.hpp"
#include <spdlog/spdlog.h>
int main(int argc, char** argv)
{
    if (argc != 5)
    {
        printf("Usage: %s <task> <engine_file> <image_path> <out_path>\n", argv[0]);
        return -1;
    }
    std::string task = argv[1];
    std::string engine_file = argv[2];
    std::string image_path = argv[3];
    std::string out_path = argv[4];
    yoloinfer::TaskType task_type;
    if (task == "det") {
        task_type = yoloinfer::TaskType::det;
    } else if (task == "seg") {
        task_type = yoloinfer::TaskType::seg;
    } else if (task == "pose") {
        task_type = yoloinfer::TaskType::pose;
    } else if (task == "obb") {
        task_type = yoloinfer::TaskType::obb;
    } else {
        printf("task type error\n");
        return -1;
    }

    float score_thr = 0.5f;
    cv::Mat src = cv::imread(image_path);

    yoloinfer::Image image(src.cols, src.rows, yoloinfer::ImageFmt::BGR);
    memcpy(image.data(yoloinfer::Device::cpu), src.data, src.total() * src.elemSize());

    yoloinfer::YoloParam param;
    param.model_path = engine_file;
    param.model_type = yoloinfer::ModelType::v8;
    param.score_thresh = score_thr;
    param.task_type = task_type;

    auto yolo = yoloinfer::CreateYoloInfer(param);
    auto outputs = yolo->forward(image);
    spdlog::info("detect size:{}", outputs.size());
    for(int i = 0; i < outputs.size(); i++) {
        const auto& box = outputs[i];
        spdlog::info("label: {}, score: {}, rect: {} {} {} {}", box.label, box.score, box.rect.x, box.rect.y, box.rect.w, box.rect.h);
        if(task_type == yoloinfer::TaskType::obb) {
            cv::RotatedRect rotate_rect;
            rotate_rect.angle =  (box.rect.angle * 180 / CV_PI);
            rotate_rect.center.x = box.rect.x;
            rotate_rect.center.y = box.rect.y;
            rotate_rect.size.width = box.rect.w;
            rotate_rect.size.height = box.rect.h;
            
            cv::Point2f vertices[4];
            rotate_rect.points(vertices);

            for (int i = 0; i < 4; i++) {
                cv::line(src, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }
        }
        else if (task_type == yoloinfer::TaskType::seg) {
            cv::rectangle(src, cv::Rect(box.rect.x, box.rect.y, box.rect.w, box.rect.h), cv::Scalar(0, 255, 0), 2);
            if(box.seg) {
                std::string seg_path = std::to_string(i) + ".png";
                cv::Mat mask(box.seg->height(), box.seg->width(), CV_8UC1, box.seg->data(yoloinfer::Device::cpu));
                cv::imwrite(seg_path, mask*255);
            }
        } else {
            cv::rectangle(src, cv::Rect(box.rect.x, box.rect.y, box.rect.w, box.rect.h), cv::Scalar(0, 255, 0), 2);
            for(const auto& point : box.points) {
                cv::circle(src, cv::Point(point.x, point.y), 4, cv::Scalar(0, 0, 255), -1);
            }
        }
    }

    cv::imwrite(out_path, src);
    return 0;
}