
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <thread>

//#include <cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <eigen3/Eigen/Dense>
#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 2;
string sData_path = "/code/cpp/EuRoC_data/V1_03_difficult/mav0/";
string sConfig_path = "../config/";

std::shared_ptr<System> pSystem;

void PubImuData() {
    string sImu_data_file = sData_path + "imu0/data.csv";
    cout << "1 PubImuData start sImu_data_file: " << sImu_data_file << endl;

    // 读取imu数据
    ifstream fsImu;
    fsImu.open(sImu_data_file.c_str());
    if (!fsImu.is_open()) {
        cerr << "Failed to open imu file! " << sImu_data_file << endl;
        return;
    }

    std::string sImu_line; //行数据的临时存储
    string dStampNSec, vGyr_x, vGyr_y, vGyr_z, vAcc_x, vAcc_y, vAcc_z; // 行数据解析的临时存储
    Eigen::Vector3d vGyr; // 陀螺仪数据的临时存储
    Eigen::Vector3d vAcc; // 加速度计数据的临时存储

    std::getline(fsImu, sImu_line); // 从fsImu读取一行数据到sImu_line，不做任何处理，跳过表头

    while (std::getline(fsImu, sImu_line) && !sImu_line.empty())
    {
        // 解析行数据
        stringstream ss(sImu_line);
        std::getline(ss, dStampNSec, ',');
        std::getline(ss, vGyr_x, ',');
        std::getline(ss, vGyr_y, ',');
        std::getline(ss, vGyr_z, ',');
        std::getline(ss, vAcc_x, ',');
        std::getline(ss, vAcc_y, ',');
        std::getline(ss, vAcc_z, ',');
        // 字符串转化为long double
        vGyr.x() = stold(vGyr_x);
        vGyr.y() = stold(vGyr_y);
        vGyr.z() = stold(vGyr_z);
        vAcc.x() = stold(vAcc_x);
        vAcc.y() = stold(vAcc_y);
        vAcc.z() = stold(vAcc_z);

        pSystem->PubImuData(stod(dStampNSec) / 1e9, vGyr, vAcc); // 将数据传入pSystem(System实例)
        usleep(5000 * nDelayTimes); // 等待
    }
    fsImu.close();
}

void PubImageData() {
    string sImage_file = sData_path + "cam0/data.csv";
    cout << "1 PubImageData start sImage_file: " << sImage_file << endl;

    // 读取图片文件名
    ifstream fsImage;
    fsImage.open(sImage_file);
    if (!fsImage.is_open()) {
        cerr << "Failed to open image file! " << sImage_file << endl;
        return;
    }

    std::string sImage_line;            // 行数据的临时存储
    string dStampNSec, sImgFileName;    // 时间戳和图片名的临时存储
    std::getline(fsImage, sImage_line); // 从fsImage读取一行数据到sImage_line，不做任何处理，跳过表头
    while (std::getline(fsImage, sImage_line) && !sImage_line.empty()) {

        stringstream ss(sImage_line);
        std::getline(ss, dStampNSec, ',');
        std::getline(ss, sImgFileName, ',');
        sImgFileName.pop_back();         // 移除sImgFileName的最后一个字符

        string imagePath = sData_path + "cam0/data/" + sImgFileName;
        Mat img = imread(imagePath.c_str(), 0); // 根据路径和文件名，用openCV读取为CV::Mat格式数据
        if (img.empty()) {
            cerr << "image is empty! path: " << imagePath << endl;
            return;
        }
        pSystem->PubImageData(stod(dStampNSec) / 1e9, img); // 将数据传入pSystem(System实例)
        // cv::imshow("SOURCE IMAGE", img);
        // cv::waitKey(0);
        usleep(50000 * nDelayTimes);
    }
    fsImage.close();
}


int main(int argc, char **argv) {
  if (argc != 3) {
    cerr << "./run_euroc PATH_TO_FOLDER/MH-05/mav0 PATH_TO_CONFIG/config \n"
         << "For example: ./run_euroc /home/stevencui/dataset/EuRoC/MH-05/mav0/ ../config/" << endl;
    return -1;
  }
  sData_path = argv[1];
  sConfig_path = argv[2];

  pSystem.reset(new System(sConfig_path));

  std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);

  // sleep(5);
  std::thread thd_PubImuData(PubImuData);

  std::thread thd_PubImageData(PubImageData);

  thd_PubImuData.join();
  thd_PubImageData.join();

// thd_BackEnd.join();

  cout << "main end... see you ..." << endl;
  return 0;
}
