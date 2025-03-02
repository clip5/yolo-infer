#pragma once
#include <future>
#include "common.hpp"

namespace yoloinfer {
    
    
    class Allocator;
    class Buffer
    {
    public:
        Buffer() = default;
        Buffer(unsigned long long size);
        Buffer& operator=(const Image& img);
        ~Buffer() = default;
        
        void* data(const Device& device) const;
        void syn(Device d1, Device d2) const;
        unsigned long long size() const {return size_;};

    private:
        std::shared_ptr<Allocator> alloc_;
        unsigned long long size_{};
        
    };

    class Tensor
    {
    public:
        enum class Type
        {
            UINT8 = 0,
            FP8 = 1,
            FP16 = 2,
            FP32 = 3,
            FP64
        };

        Tensor(std::vector<uint32_t> shape, Type type = Type::UINT8);
        Tensor() = default;
        ~Tensor() = default;
        Tensor& operator=(const Image& img);

        void* data(const Device& device) const;
        void syn(Device d1, Device d2) const;

        Type type() const { return type_; }
        std::vector<uint32_t> shape() const;
        unsigned long long size();
        unsigned long long bytes();

    private:
        std::shared_ptr<Allocator> alloc_; // allocator
        Type type_{};
        std::vector<uint32_t> shape_;
        unsigned long long size_{};
        unsigned long long bytes_{};
    };

}