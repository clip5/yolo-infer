#pragma once
#include <vector>
#include <map>
#include <mutex>
#include <future>
#include <iostream>
#include <string.h>
#include "stream.hpp"
#include "yi_type.hpp"
#include "spdlog/spdlog.h"

namespace yoloinfer
{
    using Config = std::map<std::string, std::string>;
    // using Stream = void*;

    #define GPU_BLOCK_THREADS 512

    static inline unsigned int grid_dims(unsigned int numJobs)
    {
        int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
        return ((numJobs + numBlockThreads - 1) / (float)numBlockThreads);
    }

    static inline unsigned int block_dims(unsigned int numJobs)
    {
        return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    }

    

    enum class NormType : int
    {
        None = 0,
        MeanStd = 1,  // out = (x * alpha - mean) / std
        AlphaBeta = 2 // out = x * alpha + beta
    };

    // 设置输入通道是RGB还是BGR
    enum class ChannelType : int
    {
        BGR = 0,
        RGB = 1
    };

    // 可以通过该结构体来初始化对输入的配置
    struct Norm
    {
        float mean[3];
        float std[3];
        float alpha, beta;
        NormType type = NormType::None;
        ChannelType channel_type = ChannelType::BGR;

        // out = (x * alpha - mean) / std
        static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f, ChannelType channel_type = ChannelType::BGR)
        {
            Norm out;
            out.type = NormType::MeanStd;
            out.alpha = alpha;
            out.channel_type = channel_type;
            memcpy(out.mean, mean, sizeof(out.mean));
            memcpy(out.std, std, sizeof(out.std));
            return out;
        };
        // out = x * alpha + beta
        static Norm alpha_beta(float alpha, float beta = 0.0f, ChannelType channel_type = ChannelType::BGR)
        {
            Norm out;
            out.type = NormType::AlphaBeta;
            out.alpha = alpha;
            out.beta = beta;
            out.channel_type = channel_type;
            return out;
        };
        // None
        static Norm None()
        {
            return Norm();
        }
    };

    

   
    
    
} // namespace yoloinfer
