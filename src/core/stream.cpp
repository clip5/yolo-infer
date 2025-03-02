#include "stream.hpp"
#include <cuda_runtime.h>

namespace yoloinfer
{
    Stream::Stream()
    {
        cudaStreamCreate((cudaStream_t*)&stream_);
    }

    Stream::~Stream()
    {
        cudaStreamDestroy((cudaStream_t)stream_);
    }
    
    int Stream::sync()
    {
        return cudaStreamSynchronize((cudaStream_t)stream_);
    }
    void* Stream::get()
    {
        return stream_;
    }

    void stream_sync(void* stream)
    {
        cudaStreamSynchronize((cudaStream_t)stream);
    }
}