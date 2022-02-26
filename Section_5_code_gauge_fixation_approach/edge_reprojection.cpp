#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include "vertex_pose.h"
#include "vertex_point_xyz.h"
#include "edge_reprojection.h"
#include "eigen_types.h"

#include <iostream>

namespace myslam {
    namespace backend {

        /*    std::vector<std::shared_ptr<Vertex>> verticies_; // �ñ߶�Ӧ�Ķ���
            VecX residual_;                 // �в�
            std::vector<MatXX> jacobians_;  // �ſɱȣ�ÿ���ſɱ�ά���� residual x vertex[i]
            MatXX information_;             // ��Ϣ����
            VecX observation_;              // �۲���Ϣ
            */

        void EdgeReprojection::ComputeResidual() {
            double inv_dep_i = verticies_[0]->Parameters()[0];  //��ǰ������������
            //verticies_ ��ǰ�߶�Ӧ�����ж��� defined in edge.h

            VecX param_i = verticies_[1]->Parameters();         // ��0֡��λ�� from camera frame to world frame
            Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
            Vec3 Pi = param_i.head<3>();

            VecX param_j = verticies_[2]->Parameters();         // ��i֡��λ�� from camera frame to world frame
            Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
            Vec3 Pj = param_j.head<3>();

            Vec3 pts_camera_i = pts_i_ / inv_dep_i;
            // pts_i_: camera ��0�����/��0֡ in camera frame �� ��ǰ������Ĺ۲�ֵ
            // pts_camera_i: ��ǰ�������ڵ�0�����/��0֡�� camera frame �Ŀռ�����
            Vec3 pts_imu_i = qic * pts_camera_i + tic;
            // qic: ��ת from camera frame to imu frame
            // tic: ƽ�� from camera frame to imu frame
            // pts_imu_i: ��ǰ�������ڵ�0�����/��0֡�� imu frame �Ŀռ�����
            Vec3 pts_w = Qi * pts_imu_i + Pi;
            // pts_w: ��ǰ�������� world frame �Ŀռ�����
            Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);
            // pts_imu_j: ��ǰ�������ڵ�j�����/��j֡�� imu frame �Ŀռ�����
            Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);
            // pts_camera_j: ��ǰ�������ڵ�j�����/��j֡�� camera frame �Ŀռ�����

            double dep_j = pts_camera_j.z();
            residual_ = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();   /// J^t * J * delta_x = - J^t * r
            // (pts_camera_j / dep_j).head<2>(): ��j�����/��j֡ in camera frame �� ��ǰ�����������ֵ/Ԥ��ֵ
            // pts_j_.head<2>(): ��j�����/��j֡ in camera frame �� ��ǰ������Ĺ۲�ֵ
            // residual_ = information_ * residual_;   // remove information here, we multi information matrix in problem solver
        }

        void EdgeReprojection::SetTranslationImuFromCamera(Eigen::Quaterniond& qic_, Vec3& tic_) {
            qic = qic_; // qic: ��ת from camera frame to imu frame
            tic = tic_; // tic: ƽ�� from camera frame to imu frame
        }

        void EdgeReprojection::ComputeJacobians() {
            double inv_dep_i = verticies_[0]->Parameters()[0];

            VecX param_i = verticies_[1]->Parameters();  //��0֡λ��
            Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
            Vec3 Pi = param_i.head<3>();

            VecX param_j = verticies_[2]->Parameters();  //��j֡λ��
            Qd Qj(param_j[6], param_j[3], param_j[4], param_j[5]);
            Vec3 Pj = param_j.head<3>();

            Vec3 pts_camera_i = pts_i_ / inv_dep_i;
            Vec3 pts_imu_i = qic * pts_camera_i + tic;
            Vec3 pts_w = Qi * pts_imu_i + Pi;
            Vec3 pts_imu_j = Qj.inverse() * (pts_w - Pj);
            Vec3 pts_camera_j = qic.inverse() * (pts_imu_j - tic);

            double dep_j = pts_camera_j.z();

            Mat33 Ri = Qi.toRotationMatrix();
            Mat33 Rj = Qj.toRotationMatrix();
            Mat33 ric = qic.toRotationMatrix();
            Mat23 reduce(2, 3);
            reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
                0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
            //    reduce = information_ * reduce;

            Eigen::Matrix<double, 2, 6> jacobian_pose_i;
            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Sophus::SO3d::hat(pts_imu_i);
            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

            Eigen::Matrix<double, 2, 6> jacobian_pose_j;
            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_j);
            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;

            Eigen::Vector2d jacobian_feature;
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_ * -1.0 / (inv_dep_i * inv_dep_i);

            jacobians_[0] = jacobian_feature;
            jacobians_[1] = jacobian_pose_i;
            jacobians_[2] = jacobian_pose_j;

            ///------------- check jacobians -----------------
        //    {
        //        std::cout << jacobians_[0] <<std::endl;
        //        const double eps = 1e-6;
        //        inv_dep_i += eps;
        //        Eigen::Vector3d pts_camera_i = pts_i_ / inv_dep_i;
        //        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
        //        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
        //        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
        //        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
        //
        //        Eigen::Vector2d tmp_residual;
        //        double dep_j = pts_camera_j.z();
        //        tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j_.head<2>();
        //        tmp_residual = information_ * tmp_residual;
        //        std::cout <<"num jacobian: "<<  (tmp_residual - residual_) / eps <<std::endl;
        //    }

        }

        void EdgeReprojectionXYZ::ComputeResidual() {
            Vec3 pts_w = verticies_[0]->Parameters();

            VecX param_i = verticies_[1]->Parameters();
            Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
            Vec3 Pi = param_i.head<3>();

            Vec3 pts_imu_i = Qi.inverse() * (pts_w - Pi);
            Vec3 pts_camera_i = qic.inverse() * (pts_imu_i - tic);

            double dep_i = pts_camera_i.z();
            residual_ = (pts_camera_i / dep_i).head<2>() - obs_.head<2>();
        }

        void EdgeReprojectionXYZ::SetTranslationImuFromCamera(Eigen::Quaterniond& qic_, Vec3& tic_) {
            qic = qic_;
            tic = tic_;
        }

        void EdgeReprojectionXYZ::ComputeJacobians() {

            Vec3 pts_w = verticies_[0]->Parameters();

            VecX param_i = verticies_[1]->Parameters();
            Qd Qi(param_i[6], param_i[3], param_i[4], param_i[5]);
            Vec3 Pi = param_i.head<3>();

            Vec3 pts_imu_i = Qi.inverse() * (pts_w - Pi);
            Vec3 pts_camera_i = qic.inverse() * (pts_imu_i - tic);

            double dep_i = pts_camera_i.z();

            Mat33 Ri = Qi.toRotationMatrix();
            Mat33 ric = qic.toRotationMatrix();
            Mat23 reduce(2, 3);
            reduce << 1. / dep_i, 0, -pts_camera_i(0) / (dep_i * dep_i),
                0, 1. / dep_i, -pts_camera_i(1) / (dep_i * dep_i);

            Eigen::Matrix<double, 2, 6> jacobian_pose_i;
            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * -Ri.transpose();
            jaco_i.rightCols<3>() = ric.transpose() * Sophus::SO3d::hat(pts_imu_i);
            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;

            Eigen::Matrix<double, 2, 3> jacobian_feature;
            jacobian_feature = reduce * ric.transpose() * Ri.transpose();

            jacobians_[0] = jacobian_feature;
            jacobians_[1] = jacobian_pose_i;

        }

        void EdgeReprojectionPoseOnly::ComputeResidual() {
            VecX pose_params = verticies_[0]->Parameters();
            Sophus::SE3d pose(
                Qd(pose_params[6], pose_params[3], pose_params[4], pose_params[5]),
                pose_params.head<3>()
            );

            Vec3 pc = pose * landmark_world_;
            pc = pc / pc[2];
            Vec2 pixel = (K_ * pc).head<2>() - observation_;
            // TODO:: residual_ = ????
            residual_ = pixel;
        }

        void EdgeReprojectionPoseOnly::ComputeJacobians() {
            // TODO implement jacobian here
        }

    }
}