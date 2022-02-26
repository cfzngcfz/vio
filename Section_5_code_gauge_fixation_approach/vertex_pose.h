#ifndef MYSLAM_BACKEND_POSEVERTEX_H
#define MYSLAM_BACKEND_POSEVERTEX_H

#include <memory>
#include "vertex.h"

namespace myslam {
    namespace backend {

        /**
         * Pose vertex
         * parameters: tx, ty, tz, qx, qy, qz, qw, 7 DoF
         * optimization is perform on manifold, so update is 6 DoF, left multiplication
         *
         * pose is represented as Twb in VIO case
         */
        class VertexPose : public Vertex { // 继承"vertex.h"中的class Vertex
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            VertexPose() : Vertex(7, 6) {} //自变量的维度是7，本地优化参数维度是6

            /// 加法，可重定义
            /// 默认是向量加
            virtual void Plus(const VecX& delta) override; // override 表示派生类重写基类虚函数 Plus

            std::string TypeInfo() const {
                return "VertexPose";
            }

            /**
             * 需要维护[H|b]矩阵中的如下数据块
             * p: pose, m:mappoint
             *
             *     Hp1_p2
             *     Hp2_p2    Hp2_m1    Hp2_m2    Hp2_m3     |    bp2
             *
             *                         Hm2_m2               |    bm2
             *                                   Hm2_m3     |    bm3
             * 1. 若该Camera为source camera，则维护vHessionSourceCamera；
             * 2. 若该Camera为measurement camera, 则维护vHessionMeasurementCamera；
             * 3. 并一直维护m_HessionDiagonal；
             */
        };

    }
}

#endif
