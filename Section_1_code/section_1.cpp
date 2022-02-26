#include <iostream>
#include<Eigen/Core> // Eigen 核心部分
#include<Eigen/Dense> // 稠密矩阵的代数运算（逆，特征值等）
#include<Eigen/Geometry> //几何模块（旋转、平移等）
using namespace std;
#include "sophus/se3.hpp"


int main()
{
    double theta = M_PI / 6;
    Eigen::Vector3d rotation_axis(1 / sqrt(6), 1 / sqrt(3), 1 / sqrt(2));
    Eigen::AngleAxisd rotation_vector(theta, rotation_axis); // 旋转向量
    Eigen::Matrix3d rotation_matrix = rotation_vector.matrix(); // 旋转矩阵
    cout << "更新前:\n" << rotation_matrix << endl;

    // cout << "旋转矩阵的行列式 = " << rotation_matrix.determinant() << endl;
    // cout << "验证旋转矩阵是正交矩阵" << rotation_matrix * rotation_matrix.transpose() << endl;

    // 第一章作业2 - 右乘扰动模型
    // 1.右乘李代数
    Eigen::Vector3d detla_rho(0.01, 0.02, 0.03);
    cout.precision(9); //保留9位小数
    Eigen::Matrix3d update_rotation_matrix = rotation_matrix * Sophus::SO3d::exp(detla_rho).matrix();
    cout << "李代数更新:\n" << update_rotation_matrix << endl;

    // 2.右乘四元数
    // cout << "扰动量的 norm = " << Eigen::Quaterniond(1, 0.01 / 2, 0.02 / 2, 0.03 / 2).norm() << endl;
    // cout << "扰动量归一化后的 norm = " << Eigen::Quaterniond(1, 0.01 / 2, 0.02 / 2, 0.03 / 2).normalized().norm() << endl; // 注意扰动量的归一化处理
    Eigen::Matrix3d update_quaternion = (Eigen::Quaterniond(rotation_matrix) * (Eigen::Quaterniond(1, 0.01 / 2, 0.02 / 2, 0.03 / 2).normalized())).matrix();
    cout << "四元数更新:\n" << update_quaternion << endl; //问题为什么用向量[1,detla_rho/2]更新，而不是向量[0,detla_rho/2]，dq = q*[0,detla_rho/2]转置吗？

    // 3.两者区别
    cout << "different:\n" << update_rotation_matrix - update_quaternion << endl;

    return 0;
}
