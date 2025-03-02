#pragma once

namespace yoloinfer
{
    enum class DataType
    {
        UINT8 = 0,
        HALF = 1,
        FLOAT = 2,
        INT32 = 3,
        FLOAT64 = 4,
    };

    enum class Device
    {
        cpu = 0,
        gpu = 1
    };

    enum class ModelType
    {
        v5 = 0,
        x = 1,
        v6 = 2,
        v7 = 3,
        v8 = 4,
        v9 = 5,
        v10 = 6,
        v12 = 7,
    };

    enum class TaskType
    {
        det = 0,
        seg = 1,
        pose = 2,
        obb = 3
    };

    enum class ImageFmt 
    {
        Y = 0,
        YUV420 = 1,
        NV12 = 2,
        NV21 = 3,
        RGB = 4,
        BGR = 5,
        RGBA = 6,
        BGRA = 7
    };

    class Allocator;
    class Buffer;
    class Tensor;
    class Image 
    {
    public:
        Image() = default;
        Image(int width, int height, ImageFmt format = ImageFmt::Y, DataType type = DataType::UINT8);
        ~Image();
        int width() const {return m_width;};
        int height() const {return m_height;};
        int size() const {return m_size;};
        int channels() const {return m_channels;};
        DataType type() const {return m_type;};
        ImageFmt fmt() const {return m_format;};

        void* data(const Device& device) const;
        void syn(Device d1, Device d2) const;
        
    private:
        int m_width{}, m_height{}, m_size{};
        int m_channels{1};
        DataType m_type{};
        ImageFmt m_format{};
        std::shared_ptr<Allocator> m_allocator;
        friend class Buffer;
        friend class Tensor;
    };

    struct Box
    {
        template <typename T>
        struct Rect {
            T x, y, w, h;
            float angle;
        };

        template <typename T>
        struct Point {
            T x, y;
        };
        
        float score{};
        unsigned int label{};
        Rect<float> rect{};
        std::vector<Point<float>> points{}; // for pose task
        std::shared_ptr<Image> seg;     // for seg task
        Box(const Rect<float>& rect_, float score_, unsigned int label_) 
            : rect(rect_), score(score_), label(label_) {};

        Box(const Rect<float>& rect_, const std::vector<Point<float>>& points_, float score_, unsigned int label_) 
            : rect(rect_), points(points_), score(score_), label(label_) {};
        Box() = default;
    };
}
