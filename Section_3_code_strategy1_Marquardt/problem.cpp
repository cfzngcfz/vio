#include <iostream>
#include <fstream>
#include <Eigen/Dense>
//#include <glog/logging.h>
#include "problem.h"
#include "tic_toc.h"

#ifdef USE_OPENMP

#include <omp.h>

#endif

using namespace std;

// ������С��������

namespace myslam {
    namespace backend {
        void Problem::LogoutVectorSize() {
            // LOG(INFO) <<
            //           "1 problem::LogoutVectorSize verticies_:" << verticies_.size() <<
            //           " edges:" << edges_.size();
        }

        Problem::Problem(ProblemType problemType) :
            problemType_(problemType) {
            LogoutVectorSize();
            verticies_marg_.clear();
        }

        Problem::~Problem() {}

        bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) {
            if (verticies_.find(vertex->Id()) != verticies_.end()) {
                // LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
                return false;
            }
            else {
                verticies_.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex));
            }

            return true;
        }



        bool Problem::AddEdge(shared_ptr<Edge> edge) {
            if (edges_.find(edge->Id()) == edges_.end()) {
                edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
            }
            else {
                // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
                return false;
            }

            for (auto& vertex : edge->Verticies()) {
                vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
            }
            return true;
        }


        bool Problem::Solve(int iterations) {

            if (edges_.size() == 0 || verticies_.size() == 0) {
                std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
                return false;
            }

            TicToc t_solve;
            // ͳ�ƴ��Ż�������ά����Ϊ���� H ������׼��
            SetOrdering();
            // ����edge, ���� H = J^T * J ����
            MakeHessian();
            // LM ��ʼ��
            ComputeLambdaInitLM(); // ������������mu�ĳ�ʼֵ
            // LM �㷨�������
            bool stop = false;
            int iter = 0;
            while (!stop && (iter < iterations)) {
                std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
                // currentChi_: ��ǰ����ʧ����ֵ; currentLambda_: ��ǰ����������
                
                bool oneStepSuccess = false;
                int false_cnt = 0;
                while (!oneStepSuccess)  // ���ϳ��� Lambda, ֱ���ɹ�����һ��  �����������ӣ�
                {
                    // setLambda
                    AddLambdatoHessianLM(); // �Խ��߼��ϵ�ǰ����������,Ϊ�����\Delat x_lm
                    // ���Ĳ��������Է��� H X = B
                    SolveLinearSystem();     // ��� ( J_f^T * J_f + mu*I ) * \Delat x_lm = - J_f^T * f(x)
                    RemoveLambdaHessianLM(); // �Խ��߼�ȥ��ǰ���������ӣ���Ϊÿ�ε����е����������Ǳ仯��

                    // �Ż��˳�����1�� delta_x_ ��С���˳�
                    if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10) {
                        stop = true;
                        break;
                    }

                    // ����״̬�� X = X+ delta_x
                    UpdateStates();
                    // �жϵ�ǰ���Ƿ�����Լ� LM �� lambda ��ô����
                    oneStepSuccess = IsGoodStepInLM(); // �ж���ʧ�����Ƿ��½�
                    // ��������
                    if (oneStepSuccess) {
                        // �����ʧ�����½�
      
                        // �������Ի��� ���� hessian
                        MakeHessian();
                        //// TODO:: ����ж��������Զ��������� b_max <= 1e-12 ���Ѵﵽ���������ֵ������Ӧ���þ���ֵ���������ֵ
                        //double b_max = 0.0;
                        //for (int i = 0; i < b_.size(); ++i) {
                        //    b_max = max(fabs(b_(i)), b_max);
                        //}
                        //// �Ż��˳�����2�� ����в� b_max �Ѿ���С�ˣ��Ǿ��˳�
                        //stop = (b_max <= 1e-12);
                        false_cnt = 0;
                    }
                    else {
                        // �����ʧ����û���½�
                        false_cnt++;
                        RollbackStates();   // ���û�½����ع����� UpdateStates() �������
                    }
                }
                iter++;

                // �Ż��˳�����3�� currentChi_ ����һ�ε�chi2��ȣ��½��� 1e6 �����˳�
                if (sqrt(currentChi_) <= stopThresholdLM_)  // stopThresholdLM_ defined in Problem::ComputeLambdaInitLM
                    stop = true;
            }
            std::cout << "---------" << std::endl;
            std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
            std::cout << "  makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
            return true;
        }


        void Problem::SetOrdering() {

            // ÿ�����¼���
            ordering_poses_ = 0;
            ordering_generic_ = 0;
            ordering_landmarks_ = 0;

            // Note:: verticies_ �� map ���͵�, ˳���ǰ��� id �������
            // ͳ�ƴ����Ƶ����б�������ά��
            for (auto vertex : verticies_) {
                ordering_generic_ += vertex.second->LocalDimension();  // ���е��Ż�������ά��
            }
        }

        void Problem::MakeHessian() {
            TicToc t_h;
            // ֱ�ӹ����� H ����
            ulong size = ordering_generic_;
            MatXX H(MatXX::Zero(size, size));
            VecX b(VecX::Zero(size));

            // TODO:: accelate, accelate, accelate
        //#ifdef USE_OPENMP
        //#pragma omp parallel for
        //#endif

            // ����ÿ���в���������ǵ��ſ˱ȣ��õ����� H = J^T * J
            for (auto& edge : edges_) {

                edge.second->ComputeResidual();
                edge.second->ComputeJacobians();

                auto jacobians = edge.second->Jacobians();
                auto verticies = edge.second->Verticies();
                assert(jacobians.size() == verticies.size());
                for (size_t i = 0; i < verticies.size(); ++i) {
                    auto v_i = verticies[i];
                    if (v_i->IsFixed()) continue;    // Hessian �ﲻ��Ҫ���������Ϣ��Ҳ���������ſ˱�Ϊ 0

                    auto jacobian_i = jacobians[i];
                    ulong index_i = v_i->OrderingId();
                    ulong dim_i = v_i->LocalDimension();

                    MatXX JtW = jacobian_i.transpose() * edge.second->Information();
                    for (size_t j = i; j < verticies.size(); ++j) {
                        auto v_j = verticies[j];

                        if (v_j->IsFixed()) continue;

                        auto jacobian_j = jacobians[j];
                        ulong index_j = v_j->OrderingId();
                        ulong dim_j = v_j->LocalDimension();

                        assert(v_j->OrderingId() != -1);
                        MatXX hessian = JtW * jacobian_j;
                        // ���е���Ϣ�����������
                        H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                        if (j != i) {
                            // �ԳƵ�������
                            H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                        }
                    }
                    b.segment(index_i, dim_i).noalias() -= JtW * edge.second->Residual();
                }

            }
            Hessian_ = H;
            b_ = b;
            t_hessian_cost_ += t_h.toc();

            delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;

        }

        /*
        * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
        */
        void Problem::SolveLinearSystem() {

            delta_x_ = Hessian_.inverse() * b_;
            //        delta_x_ = H.ldlt().solve(b_);

        }

        void Problem::UpdateStates() {
            for (auto vertex : verticies_) {
                ulong idx = vertex.second->OrderingId();   // ��Ϊverticies_ �� HashVertex, vertex.second ָ����verticies_�ĵڶ���(value), �����Ż��Ĳ���
                ulong dim = vertex.second->LocalDimension();
                VecX delta = delta_x_.segment(idx, dim);

                // ���еĲ��� x ����һ������  x_{k+1} = x_{k} + delta_x
                vertex.second->Plus(delta);
            }
        }

        void Problem::RollbackStates() {
            for (auto vertex : verticies_) {
                ulong idx = vertex.second->OrderingId();
                ulong dim = vertex.second->LocalDimension();
                VecX delta = delta_x_.segment(idx, dim);

                // ֮ǰ���������˺�ʹ����ʧ���������ˣ�����Ӧ�ò�Ҫ��ε�����������԰�֮ǰ���ϵ�����ȥ��
                vertex.second->Plus(-delta);
            }
        }

        /// LM
        void Problem::ComputeLambdaInitLM() {
            ni_ = 2.;
            currentLambda_ = -1.;                                           // ��¼��������
            currentChi_ = 0.0;                                              // ��¼��ʧ����ֵ
            // TODO:: robust cost chi2
            for (auto edge : edges_) {                                      // edges_ ��¼ÿ�����ϵĲв��residual_
                currentChi_ += edge.second->Chi2();                         // Chi2() return residual_.transpose() * information_ * residual_
            }
            if (err_prior_.rows() > 0)
                currentChi_ += err_prior_.norm();

            stopThresholdLM_ = 1e-6 * currentChi_;                          // ��������Ϊ ����½� 1e-6 ��

            double maxDiagonal = 0;
            ulong size = Hessian_.cols();
            assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
            for (ulong i = 0; i < size; ++i) {
                maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal); // ȡHessian��������Խ���Ԫ�ص����ֵ
            }
            double tau = 1e-5;                                              // �Զ��� ����������Hessian��������Խ���Ԫ�ص����ֵ ����֮��Ĺ�ϵ
            currentLambda_ = tau * maxDiagonal;                             // ��ʼ����������
        }

        void Problem::AddLambdatoHessianLM() {
            ulong size = Hessian_.cols();
            assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
            for (ulong i = 0; i < size; ++i) {
                // Hessian_(i, i) += currentLambda_;
                Hessian_(i, i) += currentLambda_ * Hessian_(i, i);          // Marquardt ���� �޸�1-4
                // use eq��n (13) for hlm  (see "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" P4)
                // �� [J^T*W*J + �������� * diag(J^T*W*J)] ���� Hessian_ (see "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" P3 eq 13)
            }
        }

        void Problem::RemoveLambdaHessianLM() {
            ulong size = Hessian_.cols();
            assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
            // TODO:: ���ﲻӦ�ü�ȥһ������ֵ�ķ����Ӽ����������ֵ���ȳ����⣿��Ӧ�ñ������lambdaǰ��ֵ��������ֱ�Ӹ�ֵ
            for (ulong i = 0; i < size; ++i) {
                // Hessian_(i, i) -= currentLambda_;
                //Hessian_(i, i) -= currentLambda_ * Hessian_(i, i);          // ����д����
                Hessian_(i, i) /= (1.0 + currentLambda_);          // Marquardt ���� �޸�2-4
            }
        }

        bool Problem::IsGoodStepInLM() {
            // �ж��ݶ��Ƿ��½�

            // ����rho�ķ�ĸ
            double scale = 0;
            // scale = delta_x_.transpose() * (currentLambda_ * delta_x_ + b_); // \Delta x_lm^T * (�������� * \Delta x_lm^T + b)

            // Marquardt ���� �޸�3-4
            Eigen::Matrix3d diag = Eigen::Matrix3d::Zero();
            for (int ii = 0; ii < diag.rows(); ii++)
            {
                diag(ii, ii) = Hessian_(ii, ii);
            }
            scale = 0.5*delta_x_.transpose() * (currentLambda_ * diag * delta_x_ + b_);      // ��Ҫ�Ƶ�ȷ��ǰ���Ƿ�*0.5������
            // (see "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" P3 eq 16)

            scale += 1e-3;    // make sure it's non-zero :)

            // recompute residuals after update state, �� loss( x + \Delta x_{lm} )
            double tempChi = 0.0;                                   // ��¼ x + \Delta x_{lm} ��Ӧ����ʧ����/�в�ƽ����
            for (auto edge : edges_) {
                edge.second->ComputeResidual();
                tempChi += edge.second->Chi2();                     // Chi2() return residual_.transpose() * information_ * residual_
            }
            
            // ����rho (see "L3BundleAdjustment.pdf" eq 10)
            double rho = (currentChi_ - tempChi) / scale;

            // Marquardt ���� (see "L3BundleAdjustment.pdf" eq 13 �� "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" P4 4.1.1 ����1)
            // Marquardt ���� �޸�4-4
            if (rho > 0 && isfinite(tempChi))
            {
                currentLambda_ = (std::max)(currentLambda_ / 9., 1e-7);
                currentChi_ = tempChi;                              // ������ʧ����
                return true;
            }
            else
            {
                currentLambda_ = (std::min)(currentLambda_ * 9., 1e7); // ��������*11����Ϊ*7-9
                return false;
            }

            //// Nielsen ���� (see "L3BundleAdjustment.pdf" eq 13 �� "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" P4 4.1.1 ����3)
            //if (rho > 0 && isfinite(tempChi))                       // last step was good, ������½�
            //{
            //    // ���loss�½�
            //    double alpha = 1. - pow((2 * rho - 1), 3);
            //    alpha = std::min(alpha, 2. / 3.);                   // ��һ���� "L3BundleAdjustment.pdf" �� "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" ��û�У�Ŀ����Ϊ�˿���alpha����̫��
            //    double scaleFactor = (std::max)(1. / 3., alpha);
            //    currentLambda_ *= scaleFactor;
            //    ni_ = 2;
            //    currentChi_ = tempChi;                              // ������ʧ����
            //    return true;
            //}
            //else {
            //    // ���loss���
            //    currentLambda_ *= ni_;
            //    ni_ *= 2;
            //    return false;
            //}
        }

        /** @brief conjugate gradient with perconditioning
        *
        *  the jacobi PCG method
        *
        */
        VecX Problem::PCGSolver(const MatXX& A, const VecX& b, int maxIter = -1) {
            assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a square matrix");
            int rows = b.rows();
            int n = maxIter < 0 ? rows : maxIter;
            VecX x(VecX::Zero(rows));
            MatXX M_inv = A.diagonal().asDiagonal().inverse();
            VecX r0(b);  // initial r = b - A*0 = b
            VecX z0 = M_inv * r0;
            VecX p(z0);
            VecX w = A * p;
            double r0z0 = r0.dot(z0);
            double alpha = r0z0 / p.dot(w);
            VecX r1 = r0 - alpha * w;
            int i = 0;
            double threshold = 1e-6 * r0.norm();
            while (r1.norm() > threshold && i < n) {
                i++;
                VecX z1 = M_inv * r1;
                double r1z1 = r1.dot(z1);
                double belta = r1z1 / r0z0;
                z0 = z1;
                r0z0 = r1z1;
                r0 = r1;
                p = belta * p + z1;
                w = A * p;
                alpha = r1z1 / p.dot(w);
                x += alpha * p;
                r1 -= alpha * w;
            }
            return x;
        }

    }
}
