#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "../engine.hpp"

namespace yoloinfer
{
    class TrtLogger;
    class TrtEngine : public Engine
    {
    public:
        TrtEngine(const EnginePara &para);
        ~TrtEngine() = default;
        int Forward(const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs, void* stream = nullptr) final;
        std::vector<uint32_t> GetDim(unsigned int idex) override;

    private:
        std::unique_ptr<nvinfer1::IRuntime> runtime_;               // runtime must release after engine
        std::unique_ptr<nvinfer1::ICudaEngine> engine_;
        std::unique_ptr<nvinfer1::IExecutionContext> context_;
    };
} // namespace yoloinfer
