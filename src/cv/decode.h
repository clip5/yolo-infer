#pragma once

#include "common.hpp"
#include "database.hpp"
#define BLOCK_SIZE 32

namespace yoloinfer
{
    void decode_detect(const Tensor& input, const Tensor& output, const AffineMat& affine,
                                          float score_thresh, float nms_thresh, void* stream);

    void decode_detect_obb(const Tensor& input, const Tensor& output, const AffineMat& affine,
                                          float score_thresh, float nms_thresh, void* stream);

    void decode_detectv8(const Tensor& input, const Tensor& output, const AffineMat& affine,
                                          float score_thresh, float nms_thresh, void* stream);
    
    void decode_segv8(const Tensor& input_box, const Tensor& input_seg, const Tensor& output, const AffineMat& affine,
                                          float score_thresh, float nms_thresh, void* stream);

    void decode_pose(const Tensor& input, const Tensor& output, const AffineMat& affine,
                                          float score_thresh, float nms_thresh, void* stream);
    
        // 一般用于对yolov3/v5/v7/yolox的解析，如果你有其他任务模型的后处理需要cuda加速，也可写在这个地方
        // 默认一张图片最多的检测框是1024，可以通过传参或者直接修改默认参数改变
        void decode_detect_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                          float confidence_threshold, float *invert_affine_matrix,
                                          float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, void* stream);
        
        // nms的cuda实现
        void nms_kernel_invoker(float *parray, float nms_threshold, int max_objects, int NUM_BOX_ELEMENT, void* stream);

        

        // yolov8 detect后处理解析
        void decode_detect_yolov8_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                                 float confidence_threshold, float *invert_affine_matrix,
                                                 float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, void* stream);

        // yolov8 segment分支后处理
        void decode_single_mask(float left, float top, float *mask_weights, float *mask_predict,
                                int mask_width, int mask_height, unsigned char *mask_out,
                                int mask_dim, int out_width, int out_height, void* stream);

        // yolov8 pose后处理解析
        void decode_pose_yolov8_kernel_invoker(float *predict, int num_bboxes, int pose_num, int output_cdim,
                                               float confidence_threshold, float *invert_affine_matrix,
                                               float *parray, int MAX_IMAGE_BOXES, int NUM_BOX_ELEMENT, void* stream);

        // rtdetr后处理解析
        void decode_detect_rtdetr_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                                 float confidence_threshold, int scale_expand, float *parray, int MAX_IMAGE_BOXES,
                                                 int NUM_BOX_ELEMENT, void* stream);
        
}