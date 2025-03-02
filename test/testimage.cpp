#include "database.hpp"
#include "opencv2/opencv.hpp"
int main(int argc, char **argv)
{
    yoloinfer::Image img(1920, 1080);
    
    int width = img.width();
    int height = img.height();
    printf("width: %d, height: %d\n", width, height);
    
    unsigned char* data = static_cast<unsigned char*>(img.data(yoloinfer::Device::cpu));
    for(int j = 0; j < height; j++) {
        for(int i = 0; i < width; i++) {
            data[j*width + i] = j%255;
        }
    }

    cv::Mat cvImg(height, width, CV_8UC1, data);
    cv::imwrite("test.jpg", cvImg);
    return 0;
}