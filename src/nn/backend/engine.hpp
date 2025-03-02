#pragma once
#include <vector>
#include <map>
#include "database.hpp"

namespace yoloinfer
{
    class Engine
    {
    public:
        struct EnginePara {
            std::string model_path;
            bool bfp16{};
            bool bint8{};
            void* stream{};
        };
    public:
        Engine() = default;
        virtual ~Engine() = default;
        virtual int Forward(const std::vector<Tensor>& inputs, const std::vector<Tensor>& outputs, void* stream) = 0;
        virtual std::vector<uint32_t> GetDim(unsigned int idex) = 0;
    };
    
    std::shared_ptr<Engine> CreateEngine(const Engine::EnginePara& para);
} // namespace yoloinfer
