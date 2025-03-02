#pragma once
#include "common.hpp"

namespace yoloinfer
{
    class Allocator
    {
    public:
        Allocator(unsigned long long size);
        ~Allocator();

        void* data(Device d);
        void syn(Device d1, Device d2);
        unsigned long long size() const { return data_size_; }

    private:
        void *data_device_{};
        void *data_host_{};
        unsigned long long data_size_{0};
    };
}
