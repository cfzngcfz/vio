// 针对单个特征点的三角化
#include <iostream>
#include <vector>
#include <random>  
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

struct Pose
{
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t) :Rwc(R), qwc(R), twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    Eigen::Vector2d uv;    // 单个特征点在当前相机/帧的二维归一化像素
};
int main()
{
    int poseNums = 10;
    double radius = 8;
    std::vector<Pose> camera_pose;
    for (int n = 0; n < poseNums; ++n) {
        double theta = n * 2 * M_PI / (poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()); // 旋转向量->旋转矩阵
        // R: 第i个相机/第i帧的旋转 from camera frame to world frame
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        // t: 第i个相机/第i帧的平移 from camera frame to world frame
        camera_pose.push_back(Pose(R, t));
    }

    // 随机生成 单个特征点的三维空间坐标 in world frame
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1./1000.); // (均值，标准差) 加入噪声修改1-2
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);

    Eigen::Vector3d Pw(tx, ty, tz); //特征点在world frame中的空间坐标
    // 该特征点从第三个相机/帧开始被观测，i in [start_frame_id, end_frame_id)
    int start_frame_id = 3;
    // int end_frame_id = poseNums;
    int end_frame_id = start_frame_id + 7;
    for (int i = start_frame_id; i < end_frame_id; ++i) {
        Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
        Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc); //该特征点在第i个 camera frame 的空间坐标

        double x = Pc.x();
        double y = Pc.y();
        double z = Pc.z();

        //camera_pose[i].uv = Eigen::Vector2d(x / z, y / z); // 该特征点在第i个相机/帧的观测值(二维归一化像素)
        double u = x / z + noise_pdf(generator);    // 加入噪声修改2-2
        double v = y / z + noise_pdf(generator);    // 加入噪声修改2-2
        camera_pose[i].uv = Eigen::Vector2d(u, v);  // 加入噪声修改2-2
    }


    // 遍历所有的观测数据，并三角化
    Eigen::Vector3d P_est;           // 记录自变量y的最优值
    P_est.setZero();

    Eigen::MatrixXd D; //系数矩阵
    D.setZero(2*(end_frame_id - start_frame_id),4);
    //Eigen::Matrix<double, 2*(end_frame_id - start_frame_id), 4> D; 
    //报错,在声明Matrix时，矩阵的维度无法通过变量进行定义

    for (int k = start_frame_id; k < end_frame_id; k++) {
        Eigen::Matrix<double, 3, 4> Pk; // [Rcw, tcw]
        Pk << camera_pose[k].Rwc.transpose(), -camera_pose[k].Rwc.transpose()*camera_pose[k].twc;
        D.block(2 * (k- start_frame_id), 0, 1, 4).noalias() 
            = Pk.block(2, 0, 1, 4) * camera_pose[k].uv(0) - Pk.block(0, 0, 1, 4);
        D.block(2 * (k - start_frame_id) + 1, 0, 1, 4).noalias() 
            = Pk.block(2, 0, 1, 4) * camera_pose[k].uv(1) - Pk.block(1, 0, 1, 4);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd1(D.transpose() * D, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Vector4d singularValues = svd1.singularValues();
    std::cout << "singularValues = " << singularValues.transpose() << std::endl;
    std::cout << "sigma4/sigma3 = " << singularValues(3) / singularValues(2) << std::endl;

    Eigen::Vector4d y = svd1.matrixU().block(0, 3, 4, 1); // SVD分解的U矩阵的第4列, 是V矩阵的第4行吗!!!!
    if (y(3) != 0)
    {
        P_est(0) = y(0) / y(3);
        P_est(1) = y(1) / y(3);
        P_est(2) = y(2) / y(3);
    }
    
    
    // 进一步判断三角化结果好坏
    if (singularValues(3) / singularValues(2) < 1e-3)
        {
            std::cout << "The smallest singular value is small enough." << std::endl;
            std::cout << "ground truth = " << Pw.transpose() << std::endl;
            std::cout << "forecast result = " << P_est.transpose() << std::endl;
        }
    else {
        std::cout << "==== Rescale for D ====" << std::endl;
        //系数矩阵D中的最大值
        double max_D = 0; // or -1e10
        for (int ii = 0; ii < D.rows(); ii++)
        {
            for (int jj = 0; jj < D.cols(); jj++)
            {
                max_D = std::max(max_D, D(ii, jj));
            }
        }
        Eigen::Matrix4d S = Eigen::Matrix4d::Identity() / max_D;
        Eigen::MatrixXd D_hat; //D*S
        D_hat.setZero(2 * (end_frame_id - start_frame_id), 4);
        D_hat = D * S;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd2(D_hat.transpose() * D_hat, Eigen::ComputeThinU | Eigen::ComputeThinV);
        std::cout << "singular Values before rescale = " << svd1.singularValues().transpose() << std::endl;
        std::cout << "singular Values after rescale = " << svd2.singularValues().transpose() << std::endl;

        Eigen::Vector3d P_est2;
        Eigen::Vector4d y_hat = svd2.matrixU().block(0, 3, 4, 1); // U矩阵的第4列
        Eigen::Vector4d y2 = S * y_hat;
        if (y2(3) != 0)
        {
            P_est2(0) = y2(0) / y2(3);
            P_est2(1) = y2(1) / y2(3);
            P_est2(2) = y2(2) / y2(3);
        }
        std::cout << "ground truth = " << Pw.transpose() << std::endl;
        std::cout << "forecast result before rescale = " << P_est.transpose() << std::endl;
        std::cout << "forecast result after rescale = " << P_est2.transpose() << std::endl;
    }

    return 0;
}
