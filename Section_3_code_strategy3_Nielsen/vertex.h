#ifndef MYSLAM_BACKEND_VERTEX_H
#define MYSLAM_BACKEND_VERTEX_H

#include "eigen_types.h"

namespace myslam {
    namespace backend {

        /**
         * @brief ���㣬��Ӧһ��parameter block
         * ����ֵ��VecX�洢����Ҫ�ڹ���ʱָ��ά��
         */
        class Vertex {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            /**
             * ���캯��
             * @param num_dimension ��������ά��
             * @param local_dimension ���ز�����ά�ȣ�Ϊ-1ʱ��Ϊ�뱾��ά��һ��
             */
            explicit Vertex(int num_dimension, int local_dimension = -1);

            virtual ~Vertex();

            /// ���ر���ά��
            int Dimension() const;

            /// ���ر�������ά��
            int LocalDimension() const;

            /// �ö����id
            unsigned long Id() const { return id_; }

            /// ���ز���ֵ
            VecX Parameters() const { return parameters_; }

            /// ���ز���ֵ������
            VecX& Parameters() { return parameters_; }

            /// ���ò���ֵ
            void SetParameters(const VecX& params) { parameters_ = params; }

            /// �ӷ������ض���
            /// Ĭ����������
            virtual void Plus(const VecX& delta);

            /// ���ض�������ƣ���������ʵ��
            virtual std::string TypeInfo() const = 0;

            int OrderingId() const { return ordering_id_; }

            void SetOrderingId(unsigned long id) { ordering_id_ = id; };

            /// �̶��õ�Ĺ���ֵ
            void SetFixed(bool fixed = true) {
                fixed_ = fixed;
            }

            /// ���Ըõ��Ƿ񱻹̶�
            bool IsFixed() const { return fixed_; }

        protected:
            VecX parameters_;   // ʵ�ʴ洢�ı���ֵ
            int local_dimension_;   // �ֲ�������ά��
            unsigned long id_;  // �����id���Զ�����

            /// ordering id����problem��������id������Ѱ���ſɱȶ�Ӧ��
            /// ordering id����ά����Ϣ������ordering_id=6���ӦHessian�еĵ�6��
            /// ���㿪ʼ
            unsigned long ordering_id_ = 0;

            bool fixed_ = false;    // �Ƿ�̶�
        };

    }
}

#endif
