// ��Ե�������������ǻ�
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

    Eigen::Vector2d uv;    // �����������ڵ�ǰ���/֡�Ķ�ά��һ������
};
int main()
{
    int poseNums = 10;
    double radius = 8;
    std::vector<Pose> camera_pose;
    for (int n = 0; n < poseNums; ++n) {
        double theta = n * 2 * M_PI / (poseNums * 4); // 1/4 Բ��
        // �� z�� ��ת
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()); // ��ת����->��ת����
        // R: ��i�����/��i֡����ת from camera frame to world frame
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        // t: ��i�����/��i֡��ƽ�� from camera frame to world frame
        camera_pose.push_back(Pose(R, t));
    }

    // ������� �������������ά�ռ����� in world frame
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1./1000.); // (��ֵ����׼��) ���������޸�1-2
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);

    Eigen::Vector3d Pw(tx, ty, tz); //��������world frame�еĿռ�����
    // ��������ӵ��������/֡��ʼ���۲⣬i in [start_frame_id, end_frame_id)
    int start_frame_id = 3;
    // int end_frame_id = poseNums;
    int end_frame_id = start_frame_id + 7;
    for (int i = start_frame_id; i < end_frame_id; ++i) {
        Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
        Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc); //���������ڵ�i�� camera frame �Ŀռ�����

        double x = Pc.x();
        double y = Pc.y();
        double z = Pc.z();

        //camera_pose[i].uv = Eigen::Vector2d(x / z, y / z); // ���������ڵ�i�����/֡�Ĺ۲�ֵ(��ά��һ������)
        double u = x / z + noise_pdf(generator);    // ���������޸�2-2
        double v = y / z + noise_pdf(generator);    // ���������޸�2-2
        camera_pose[i].uv = Eigen::Vector2d(u, v);  // ���������޸�2-2
    }


    // �������еĹ۲����ݣ������ǻ�
    Eigen::Vector3d P_est;           // ��¼�Ա���y������ֵ
    P_est.setZero();

    Eigen::MatrixXd D; //ϵ������
    D.setZero(2*(end_frame_id - start_frame_id),4);
    //Eigen::Matrix<double, 2*(end_frame_id - start_frame_id), 4> D; 
    //����,������Matrixʱ�������ά���޷�ͨ���������ж���

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

    Eigen::Vector4d y = svd1.matrixU().block(0, 3, 4, 1); // SVD�ֽ��U����ĵ�4��, ��V����ĵ�4����!!!!
    if (y(3) != 0)
    {
        P_est(0) = y(0) / y(3);
        P_est(1) = y(1) / y(3);
        P_est(2) = y(2) / y(3);
    }
    
    
    // ��һ���ж����ǻ�����û�
    if (singularValues(3) / singularValues(2) < 1e-3)
        {
            std::cout << "The smallest singular value is small enough." << std::endl;
            std::cout << "ground truth = " << Pw.transpose() << std::endl;
            std::cout << "forecast result = " << P_est.transpose() << std::endl;
        }
    else {
        std::cout << "==== Rescale for D ====" << std::endl;
        //ϵ������D�е����ֵ
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
        Eigen::Vector4d y_hat = svd2.matrixU().block(0, 3, 4, 1); // U����ĵ�4��
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
