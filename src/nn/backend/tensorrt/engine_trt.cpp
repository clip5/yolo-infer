#include "engine_trt.hpp"
#include <fstream>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <filesystem>

namespace yoloinfer {

    class TrtLogger : public nvinfer1::ILogger {
        using Severity = nvinfer1::ILogger::Severity;
    public:
        TrtLogger(Severity level = Severity::kWARNING) {
            level_ = level;
        };
        ~TrtLogger() = default;
        void log(Severity severity, const char* msg) noexcept override {
            if(severity <= level_) {
                switch (severity) {
                    case Severity::kINTERNAL_ERROR:
                        printf("INTERNAL_ERROR: %s", msg);
                        break;
                    case Severity::kERROR:
                        printf("ERROR: %s", msg);
                        break;
                    case Severity::kWARNING:
                        printf("WARNING: %s", msg);
                        break;
                    case Severity::kINFO:
                        printf("INFO: %s", msg);
                        break;
                    default:
                        break;
                }
            }
        };
    private:
        Severity level_;
    };

    TrtLogger gTrtLogger{};
    TrtEngine::TrtEngine(const EnginePara &para) {

        std::ifstream model_file(para.model_path, std::ios::binary);
        if (!model_file.is_open()) {
            throw std::runtime_error("Failed to open model file: " + para.model_path);
        }

        model_file.seekg(0, std::ifstream::end);
        auto model_size = model_file.tellg();
        model_file.seekg(0, std::ifstream::beg);

        std::vector<char> model_data(model_size);
        model_file.read(model_data.data(), model_size);

        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gTrtLogger));
        if(!runtime_) {
            return;
        }
            
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(model_data.data(), model_size));
        if(!engine_) {
            return;
        }
        context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if(!context_) {
            return;
        }
    }

    // int TrtEngine::Forward(const Buffer& inputs, Buffer& outputs, void* stream) {

    //     std::vector<void*> bindings;
    //     bindings.push_back(inputs.data<Device::gpu>());
    //     bindings.push_back(outputs.data<Device::gpu>());

    //     if (context_) {
    //         context_->enqueueV2(bindings.data(), static_cast<cudaStream_t>(stream), nullptr);
    //     }
    //     return 0;
    // }

    int TrtEngine::Forward(const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs, void* stream) {
        
        std::vector<void*> bindings;
        for(auto& input : inputs) {
            bindings.push_back(input.data(Device::gpu));
        }
        for(auto& output : outputs) {
            bindings.push_back(output.data(Device::gpu));
        }
        if (context_) {
            context_->enqueueV2(bindings.data(), static_cast<cudaStream_t>(stream), nullptr);
        }
        return 0;
    }
    std::vector<uint32_t> TrtEngine::GetDim(unsigned int idex) {
        std::vector<uint32_t> dims;
        auto dim = engine_->getBindingDimensions(idex);
        for(int i = 0; i < dim.nbDims; ++i) {
            dims.push_back(dim.d[i]);
        }
        return std::move(dims);
    }

    std::shared_ptr<Engine> CreateEngine(const Engine::EnginePara &para) {
        return std::make_shared<TrtEngine>(para);
    }
}