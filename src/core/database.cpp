#include "allocator.hpp"
#include "database.hpp"

namespace yoloinfer
{
    Image::Image(int width, int height, ImageFmt format, DataType type) : m_width(width), m_height(height), m_format(format), m_type(type) {
        unsigned int img_size = width * height;
        float format_factor = 1.0;
        float type_factor = 1.0;
        switch (m_type) {
        case DataType::FLOAT:
            type_factor = 4;
            break;
        case DataType::HALF:
            type_factor = 2;
            break;
        default:
            break;
        }
        switch (m_format) {
        case ImageFmt::YUV420:
        case ImageFmt::NV12:
        case ImageFmt::NV21:
            format_factor = 1.5;
            m_channels = 1;
            break;
        case ImageFmt::BGR:
        case ImageFmt::RGB:
            format_factor = 3;
            m_channels = 3;
            break;
        case ImageFmt::BGRA:
        case ImageFmt::RGBA:
            format_factor = 4;
            m_channels = 4;
        default:
            break;
        }
        img_size *= format_factor * type_factor;
        m_size = img_size;
        m_allocator = std::make_shared<Allocator>(img_size);
    }

    Image::~Image() {}
    
    void* Image::data(const Device& device) const
    {
        if(m_allocator)
            return m_allocator->data(device);
        else
            return nullptr;
    }
    
    void Image::syn(Device d1, Device d2) const
    {
        if(m_allocator)
            m_allocator->syn(d1, d2);
    }
    

    Buffer::Buffer(unsigned long long size)
    {
        size_ = size;
        alloc_ = std::make_shared<Allocator>(size_);
    }
    
    Buffer& Buffer::operator=(const Image& image) {
        size_ = image.m_size;
        alloc_ = image.m_allocator;
        return *this;
    }

    void Buffer::syn(Device d1, Device d2) const
    {
        if(alloc_)
            alloc_->syn(d1, d2);
    }


    void* Buffer::data(const Device& device) const
    {
        if(alloc_)
            return alloc_->data(device);
        else
            return nullptr;
    }

    
    Tensor::Tensor(std::vector<uint32_t> shape, Type type) : shape_(shape), type_(type)
    {
        if (!shape_.size())
            return;
        size_ = 1;
        for (const auto &s : shape_)
            size_ *= s;

        bytes_ = size_;
        switch (type)
        {
            case Type::UINT8:
            case Type::FP8:
                break;
            case Type::FP16:
                bytes_ *= 2;
                break;
            case Type::FP32:
                bytes_ *= 4;
                break;
            case Type::FP64:
                bytes_ *= 8;
                break;
            default:
                break;
        }

        alloc_ = std::make_shared<Allocator>(bytes_);
    }

    unsigned long long Tensor::size()
    {
        return size_;
    }
    std::vector<uint32_t> Tensor::shape() const
    {
        return shape_;
    }
    unsigned long long Tensor::bytes()
    {
        return bytes_;
    }

    void* Tensor::data(const Device& device) const
    {
        if(alloc_)
            return alloc_->data(device);
        else
            return nullptr;
    }


    void Tensor::syn(Device d1, Device d2) const {
        if(alloc_)
            alloc_->syn(d1, d2);
    };


    Tensor& Tensor::operator=(const Image& image) {
        alloc_ = image.m_allocator;
        bytes_ = image.m_size;
        return *this;
    }

}