
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

    // ��ȡimu����
    ifstream fsImu;
    fsImu.open(sImu_data_file.c_str());
    if (!fsImu.is_open()) {
        cerr << "Failed to open imu file! " << sImu_data_file << endl;
        return;
    }

    std::string sImu_line; //�����ݵ���ʱ�洢
    string dStampNSec, vGyr_x, vGyr_y, vGyr_z, vAcc_x, vAcc_y, vAcc_z; // �����ݽ�������ʱ�洢
    Eigen::Vector3d vGyr; // ���������ݵ���ʱ�洢
    Eigen::Vector3d vAcc; // ���ٶȼ����ݵ���ʱ�洢

    std::getline(fsImu, sImu_line); // ��fsImu��ȡһ�����ݵ�sImu_line�������κδ���������ͷ

    while (std::getline(fsImu, sImu_line) && !sImu_line.empty())
    {
        // ����������
        stringstream ss(sImu_line);
        std::getline(ss, dStampNSec, ',');
        std::getline(ss, vGyr_x, ',');
        std::getline(ss, vGyr_y, ',');
        std::getline(ss, vGyr_z, ',');
        std::getline(ss, vAcc_x, ',');
        std::getline(ss, vAcc_y, ',');
        std::getline(ss, vAcc_z, ',');
        // �ַ���ת��Ϊlong double
        vGyr.x() = stold(vGyr_x);
        vGyr.y() = stold(vGyr_y);
        vGyr.z() = stold(vGyr_z);
        vAcc.x() = stold(vAcc_x);
        vAcc.y() = stold(vAcc_y);
        vAcc.z() = stold(vAcc_z);

        pSystem->PubImuData(stod(dStampNSec) / 1e9, vGyr, vAcc); // �����ݴ���pSystem(Systemʵ��)
        usleep(5000 * nDelayTimes); // �ȴ�
    }
    fsImu.close();
}

void PubImageData() {
    string sImage_file = sData_path + "cam0/data.csv";
    cout << "1 PubImageData start sImage_file: " << sImage_file << endl;

    // ��ȡͼƬ�ļ���
    ifstream fsImage;
    fsImage.open(sImage_file);
    if (!fsImage.is_open()) {
        cerr << "Failed to open image file! " << sImage_file << endl;
        return;
    }

    std::string sImage_line;            // �����ݵ���ʱ�洢
    string dStampNSec, sImgFileName;    // ʱ�����ͼƬ������ʱ�洢
    std::getline(fsImage, sImage_line); // ��fsImage��ȡһ�����ݵ�sImage_line�������κδ���������ͷ
    while (std::getline(fsImage, sImage_line) && !sImage_line.empty()) {

        stringstream ss(sImage_line);
        std::getline(ss, dStampNSec, ',');
        std::getline(ss, sImgFileName, ',');
        sImgFileName.pop_back();         // �Ƴ�sImgFileName�����һ���ַ�

        string imagePath = sData_path + "cam0/data/" + sImgFileName;
        Mat img = imread(imagePath.c_str(), 0); // ����·�����ļ�������openCV��ȡΪCV::Mat��ʽ����
        if (img.empty()) {
            cerr << "image is empty! path: " << imagePath << endl;
            return;
        }
        pSystem->PubImageData(stod(dStampNSec) / 1e9, img); // �����ݴ���pSystem(Systemʵ��)
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
