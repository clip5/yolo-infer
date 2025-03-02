<div align="center">
<h2>
  yolo-infer
</h2>
</div>

This is a project run yolo on tensorrt.

### Compilation
To compile the project, follow these steps:
```bash
mkdir build && cmake ../
make -j4 install
```

### Dependencies
- TensorRT
- Opencv
- spdlog (embedded)

### Example Usage
```cpp
#include "yolo_infer.hpp"
#include "opencv2/opencv.hpp"
int main(int argc, char* argv[]) 
{
    // create yolo infer
    yoloinfer::YoloParam param;
    auto yolo = yoloinfer::CreateYoloInfer(param);
    
    // read image
    cv::Mat src = cv::imread(image_path);
    yoloinfer::Image image(src.cols, src.rows, yoloinfer::ImageFmt::BGR);
    memcpy(image.data(yoloinfer::Device::cpu), src.data, src.total() * src.elemSize());

    // forward
    auto outputs = yolo->forward(image);

    return 0;
}
```
a example of yolo inference is provided in the `example` directory.

### References:
- [tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)
- [AiInfer](https://github.com/CYYAI/AiInfer)
