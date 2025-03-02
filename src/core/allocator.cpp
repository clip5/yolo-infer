#include "allocator.hpp"
#include <iostream>
#include <cuda_runtime.h>

namespace yoloinfer
{
    Allocator::Allocator(unsigned long long size) : data_size_(size)
    {
        cudaMalloc(&data_device_, size);
        cudaHostAlloc(&data_host_, size, cudaHostAllocDefault);
    }

    Allocator::~Allocator()
    {
        if (data_device_) {
            cudaFree(data_device_);
        }
        if (data_host_) {
            cudaFreeHost(data_host_);
        }
    }

    void *Allocator::data(Device device)
    {
        if (device == Device::cpu) {
            return data_host_;
        } else if (device == Device::gpu) {
            return data_device_;
        }
        return nullptr;
    }

    void Allocator::syn(Device d1, Device d2)
    {
        if(d1 != d2) {
            if(d1 == Device::cpu && d2 == Device::gpu)
                cudaMemcpy(data_device_, data_host_, data_size_ , cudaMemcpyHostToDevice);
            else if(d1 == Device::gpu && d2 == Device::cpu)
                cudaMemcpy(data_host_, data_device_, data_size_ , cudaMemcpyDeviceToHost);
        }
    }

}
