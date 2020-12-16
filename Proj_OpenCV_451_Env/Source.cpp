#include <iostream>
#include <string>
#include "opencv2/dnn_superres.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

int main(int argc, char* argv[]) {
    std::cout << "Hi" << std::endl;

    //std::string img_path = "../../Super_Resolution/Images/input.png";
    std::string img_path = "../../Super_Resolution/Images/ayg6Jsqi2.png";
    cv::Mat img_ori = cv::imread(img_path);
    cv::imshow("Test", img_ori);
    cv::waitKey(0);

    cv::dnn_superres::DnnSuperResImpl sr;
    std::string model_path = "../../Super_Resolution/models/EDSR_x3.pb";
    sr.readModel(model_path);

    sr.setModel("edsr", 3);

    //Upsample
    cv::Mat img_new;
    sr.upsample(img_ori, img_new);
    cv::imshow("Up scale", img_new);
    cv::waitKey(0);

    return 0;
}