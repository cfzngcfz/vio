#include <iostream>
#include <vector>
#include <random>  
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#define M_PI 3.14159265358979323846 // 圆周率

// 问题描述: 单个时刻，10个相机同时对20个特征点进行观测，解视觉SLAM 里的Bundle Adjustment 问题，如Section_3_VIO_BundleAdjustment.pdf P4 公式1
// 单一时刻的BA问题，不考虑滑动窗口


struct Pose
{
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t): Rwc(R), qwc(R), twc(t) {};
    Eigen::Matrix3d Rwc; // 旋转 from camera frame to world frame
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc; // 平移 in world frame
};
int main()
{
    int featureNums = 20; // 特征点的数量
    int poseNums = 10; // 滑动窗口内的pose数量
    int diem = poseNums * 6 + featureNums * 3; //H系数矩阵的维度，每个pose有6个自由度/自变量，即旋转中的3个+平移3个，每个特征点有三个自由度，即x,y,z
    double fx = 1.; //相机焦距
    double fy = 1.;
    Eigen::MatrixXd H(diem, diem);
    H.setZero();

    std::vector<Pose> camera_pose; // 记录相机的pose信息，即旋转和平移
    double radius = 8;
    for (int n = 0; n < poseNums; ++n) {
        double theta = n * 2 * M_PI / (poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        camera_pose.push_back(Pose(R, t));
    }

    // 随机数生成三维特征点
    std::default_random_engine generator;
    std::vector<Eigen::Vector3d> points; // 记录特征点的空间位置
    for (int j = 0; j < featureNums; ++j)
    {
        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
        std::uniform_real_distribution<double> z_rand(8., 10.);
        double tx = xy_rand(generator);
        double ty = xy_rand(generator);
        double tz = z_rand(generator);

        Eigen::Vector3d Pw(tx, ty, tz); // Pw 特征点在 world frame 的空间坐标
        points.push_back(Pw);

        for (int i = 0; i < poseNums; ++i) {
            Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
            Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc); // Pc 特征点在 camera frame 的空间坐标

            double x = Pc.x();
            double y = Pc.y();
            double z = Pc.z();
            double z_2 = z * z;
            Eigen::Matrix<double, 2, 3> jacobian_uv_Pc; // 在camera frame 中，残差关于特征点的偏导
            jacobian_uv_Pc << fx / z, 0, -x * fx / z_2,
                0, fy / z, -y * fy / z_2;
            Eigen::Matrix<double, 2, 3> jacobian_Pj = jacobian_uv_Pc * Rcw; // 在world frame 中，残差关于特征点的偏导
            Eigen::Matrix<double, 2, 6> jacobian_Ti; // 残差关于滑动窗口内pose的偏导
            jacobian_Ti << -x * y * fx / z_2, (1 + x * x / z_2)* fx, -y / z * fx, fx / z, 0, -x * fx / z_2,
                -(1 + y * y / z_2) * fy, x* y / z_2 * fy, x / z * fy, 0, fy / z, -y * fy / z_2;

            H.block(i * 6, i * 6, 6, 6) += jacobian_Ti.transpose() * jacobian_Ti; // JT*J 系数矩阵block
            H.block(j * 3 + 6 * poseNums, j * 3 + 6 * poseNums, 3, 3) += jacobian_Pj.transpose() * jacobian_Pj;
            H.block(i*6,j*3 + 6*poseNums, 6,3) += jacobian_Ti.transpose() * jacobian_Pj;
            H.block(j * 3 + 6 * poseNums, i * 6, 3, 6) += jacobian_Pj.transpose() * jacobian_Ti;
        }
    }

    
    // std::cout << "===== H系数矩阵 =====" << H << std::endl;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(H);
    std::cout << "===== H系数矩阵的特征值 =====" << std::endl;
    std::cout << saes.eigenvalues() <<std::endl;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // 奇异值分解
    // https://zhuanlan.zhihu.com/p/29846048
    // https://blog.csdn.net/asd136912/article/details/79864576
    std::cout << "===== H系数矩阵的奇异值 =====" << std::endl;
    std::cout << svd.singularValues() << std::endl;

    return 0;
}
