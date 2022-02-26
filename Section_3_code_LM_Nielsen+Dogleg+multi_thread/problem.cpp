#include <iostream>
#include <fstream>
#include <Eigen/Dense>
//#include <glog/logging.h>
#include "problem.h"
#include "tic_toc.h"
#include <thread>
#include <mutex> // ������

#ifdef USE_OPENMP

#include <omp.h>

#endif

using namespace std;


// ����ʹ��LM ����Dogleg
const int optimization_method = 1; // 0: LM,   1: Dogleg
// �����Ƿ�ʹ�ü���, �Լ����ٷ�ʽ
const int acceleration_method = 1; // 0: Normal,non-acc,  1: multi-threads acc
const int num_thread = 4;

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

        void Problem::SaveCostTime(std::string filename, double SolveTime, double HessianTime)
        {
            std::ofstream save_pose;
            save_pose.setf(std::ios::fixed, std::ios::floatfield);
            save_pose.open(filename.c_str(), std::ios::app);
            // long int timeStamp = floor(time*1e9);
            save_pose << SolveTime << " "
                << HessianTime << std::endl;
            save_pose.close();
        }

        bool Problem::Solve(int iterations) {
            if (edges_.size() == 0 || verticies_.size() == 0) {
                std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
                return false;
            }

            TicToc t_solve;
            // ͳ���Ż�������ά����Ϊ���� H ������׼��
            SetOrdering();
            // ����edge, ���� H ����
            MakeHessian();

            bool stop = false;
            int iter = 0;
            double last_chi_ = 1e20;

            if (optimization_method == 0) { // LM�㷨
              // LM ��ʼ��
                ComputeLambdaInitLM();
                // LM �㷨�������
                while (!stop && (iter < iterations)) {
                    std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
                    bool oneStepSuccess = false;
                    int false_cnt = 0;
                    while (!oneStepSuccess && false_cnt < 10)  // ���ϳ��� Lambda, ֱ���ɹ�����һ��
                    {
                        AddLambdatoHessianLM(); // �Խ��߼��ϵ�ǰ����������
                        SolveLinearSystem();
                        RemoveLambdaHessianLM(); // �Խ��߼�ȥ��ǰ���������ӣ���Ϊÿ�ε����е����������Ǳ仯��

                        // ����״̬��
                        UpdateStates();
                        // �жϵ�ǰ���Ƿ�����Լ� LM �� lambda ��ô����, chi2 Ҳ����һ��
                        oneStepSuccess = IsGoodStepInLM();
                        // ��������
                        if (oneStepSuccess) {
                            // �������Ի��� ���� hessian
                            MakeHessian();
                            false_cnt = 0;
                        }
                        else {
                            false_cnt++;
                            RollbackStates();  // ���û�½����ع�
                        }
                    }
                    iter++;

                    if (last_chi_ - currentChi_ < 1e-5) {
                        std::cout << "sqrt(currentChi_) <= stopThresholdLM_" << std::endl;
                        stop = true;
                    }
                    last_chi_ = currentChi_;
                }
            }
            else if (optimization_method == 1) { // Dogleg

              // ����ʼ����ʧ����
                currentChi_ = 0.0;
                for (auto edge : edges_) {
                    // ��MakeHessian()���Ѿ�������edge.second->ComputeResidual()
                    currentChi_ += edge.second->Chi2();
                }
                if (err_prior_.rows() > 0)
                    currentChi_ += err_prior_.norm();

                radius_ = 1e4;  // ��ʼ������

                while (!stop && (iter < iterations)) {
                    std::cout << "\niter: " << iter << " , currentChi= " << currentChi_ << " , radius= " << radius_ << std::endl;
                    iter++;

                    bool oneStepSuccess = false;
                    int false_cnt = 0;
                    while (!oneStepSuccess && false_cnt < 10)  // ���ϳ��� Lambda, ֱ���ɹ�����һ��
                    {
                        // step 2.1. �����½��� ���� h_sd_
                        double numerator = b_.transpose() * b_;
                        double denominator = b_.transpose() * Hessian_ * b_;
                        double alpha_ = numerator / denominator;
                        h_sd_ = alpha_ * b_;

                        // step 2.2. ��˹ţ�ٷ� ���� h_gn_
                        // To Do: �˴�Hessian_�Ƚϴ�, ֱ������ܺ�ʱ, �ɲ��� Gauss-Newton�����
                        //h_gn_ = Hessian_.inverse() * b_;
                        h_gn_ = Hessian_.ldlt().solve(b_);

                        // 3.����h_dl ����
                        if (h_gn_.norm() <= radius_) {
                            h_dl_ = h_gn_;
                        }
                        else if (h_sd_.norm() > radius_) {
                            h_dl_ = (radius_ / h_sd_.norm()) * h_sd_;
                        }
                        else {
                            double coefficient_a = (h_gn_ - h_sd_).squaredNorm();
                            double coefficient_b = 2 * h_sd_.transpose() * (h_gn_ - h_sd_);
                            double coefficient_c = h_sd_.squaredNorm() - radius_ * radius_;
                            double beta_ = (-coefficient_b + sqrt(coefficient_b * coefficient_b - 4 * coefficient_a * coefficient_c)) / 2 / coefficient_a;

                            assert(beta_ > 0.0 && beta_ < 1.0 && "Error while computing beta");
                            h_dl_ = h_sd_ + beta_ * (h_gn_ - h_sd_);
                        }
                        delta_x_ = h_dl_;

                        UpdateStates();
                        oneStepSuccess = IsGoodStepInDogleg();
                        // ��������
                        if (oneStepSuccess)
                        {
                            MakeHessian();
                            false_cnt = 0;
                        }
                        else
                        {
                            false_cnt++;
                            RollbackStates();
                        }

                    }
                    iter++;

                    if (last_chi_ - currentChi_ < 1e-5)
                    {
                        std::cout << "sqrt(currentChi_) <= stopThresholdLM_" << std::endl;
                        stop = true;
                    }
                    last_chi_ = currentChi_;
                }
            }

            std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
            std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;

            SaveCostTime("costTime.txt", t_solve.toc(), t_hessian_cost_);

            t_hessian_cost_ = 0.;
            return true;
        }

        bool Problem::IsGoodStepInDogleg() {

            // ������ʧ����
            double tempChi = 0.0;
            for (auto edge : edges_) {
                edge.second->ComputeResidual();
                tempChi += edge.second->Chi2();
            }
            if (err_prior_.size() > 0)
                //tempChi += err_prior_.norm();
                tempChi += err_prior_.norm();


            // ����rho�ķ�ĸ
            double scale = 0.0;
            if (h_dl_ == h_gn_) {
                scale = currentChi_;
            }
            else if (h_dl_ == radius_ * b_ / b_.norm()) {
                //scale = radius_ * (2 * (alpha_ * b_).norm() - radius_) / (2 * alpha_);
                scale = radius_ * (b_.norm() - radius_ / 2.0 / alpha_);
            }
            else {
                scale = 0.5 * alpha_ * (1 - beta_) * (1 - beta_) * b_.squaredNorm()
                    + beta_ * (2 - beta_) * currentChi_;
            }

            // ����rho
            double rho_ = (currentChi_ - tempChi) / scale;


            if (rho_ > 0.75 && isfinite(tempChi)) {
                radius_ = std::max(radius_, 3 * delta_x_.norm());
            }
            else if (rho_ < 0.25) {
                radius_ = std::max(radius_ / 4, 1e-7);
                // radius_ = 0.5 * radius_; // ������
            }


            if (rho_ > 0 && isfinite(tempChi)) {
                currentChi_ = tempChi;
                return true;
            }
            else {
                return false;
            }
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
            H_all_.setZero(size, size);
            b_all_.setZero(size);

            if (acceleration_method == 0) {
                std::vector<ulong> vec_all;
                for (auto& edge : edges_) {
                    vec_all.emplace_back(edge.first);
                }
                edge_thread(vec_all);
            }
            else if (acceleration_method == 1) {
                std::vector<ulong> vec_id[num_thread];
                for (auto& edge : edges_) {
                    vec_id[edge.first % num_thread].emplace_back(edge.first);
                }
                //for (int ii=0;ii<num_thread;ii++) {
                //  cout << "\nsize of vec_id[" << ii << "] = " << vec_id[ii].size() << endl;
                //  for (ulong id : vec_id[ii]) {
                //      cout << id << ", ";
                //  }
                //}
                for (int ii = 0; ii < num_thread; ii++) {
                    std::thread sub_thread = std::thread(&Problem::edge_thread, this, vec_id[ii]);
                    sub_thread.join();
                }
            }

            Hessian_ = H_all_;
            b_ = b_all_;
            t_hessian_cost_ += t_h.toc();

            delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
        }


        void Problem::edge_thread(std::vector<ulong> vec_id) {
            for (ulong id : vec_id) {
                //if (acceleration_method == 1) {
                //    m_H_.lock();
                //}
                std::shared_ptr<Edge> edge_second = edges_[id];
                edge_second->ComputeResidual();
                edge_second->ComputeJacobians();

                // TODO:: robust cost
                auto jacobians = edge_second->Jacobians();
                auto verticies = edge_second->Verticies();
                assert(jacobians.size() == verticies.size());
                for (size_t i = 0; i < verticies.size(); ++i) {
                    auto v_i = verticies[i];
                    if (v_i->IsFixed()) continue;  // Hessian �ﲻ��Ҫ���������Ϣ��Ҳ���������ſ˱�Ϊ 0

                    auto jacobian_i = jacobians[i];
                    ulong index_i = v_i->OrderingId();
                    ulong dim_i = v_i->LocalDimension();

                    MatXX JtW = jacobian_i.transpose() * edge_second->Information();
                    for (size_t j = i; j < verticies.size(); ++j) {
                        auto v_j = verticies[j];

                        if (v_j->IsFixed()) continue;

                        auto jacobian_j = jacobians[j];
                        ulong index_j = v_j->OrderingId();
                        ulong dim_j = v_j->LocalDimension();

                        assert(v_j->OrderingId() != -1);
                        MatXX hessian = JtW * jacobian_j;

                        // ���е���Ϣ�����������
                        if (acceleration_method == 1) {
                            m_H_.lock();
                        }
                        H_all_.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                        if (j != i) {
                            // �ԳƵ�������
                            H_all_.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                        }
                        if (acceleration_method == 1) {
                            m_H_.unlock();
                        }
                    }
                    if (acceleration_method == 1) {
                        m_b_.lock();
                    }
                    b_all_.segment(index_i, dim_i).noalias() -= JtW * edge_second->Residual();
                    if (acceleration_method == 1) {
                        m_b_.unlock();
                    }
                }
                //if (acceleration_method == 1) {
                //    m_H_.unlock();
                //}
            }
        }

        /*
        * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
        */
        void Problem::SolveLinearSystem() {
            delta_x_ = Hessian_.inverse() * b_;
            //delta_x_ = H.ldlt().solve(b_);
        }

        void Problem::UpdateStates() {
            for (auto vertex : verticies_) {
                ulong idx = vertex.second->OrderingId();
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
                Hessian_(i, i) += currentLambda_;
            }
        }

        void Problem::RemoveLambdaHessianLM() {
            ulong size = Hessian_.cols();
            assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
            // TODO:: ���ﲻӦ�ü�ȥһ������ֵ�ķ����Ӽ����������ֵ���ȳ����⣿��Ӧ�ñ������lambdaǰ��ֵ��������ֱ�Ӹ�ֵ
            for (ulong i = 0; i < size; ++i) {
                Hessian_(i, i) -= currentLambda_;
            }
        }

        bool Problem::IsGoodStepInLM() {
            // �ж��ݶ��Ƿ��½�

            // ����rho�ķ�ĸ
            double scale = 0;
            scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_); // \Delta x_lm^T * (�������� * \Delta x_lm^T + b)
            scale += 1e-3;    // make sure it's non-zero :)

            // recompute residuals after update state, �� loss( x + \Delta x_{lm} )
            double tempChi = 0.0;                                   // ��¼ x + \Delta x_{lm} ��Ӧ����ʧ����/�в�ƽ����
            for (auto edge : edges_) {
                edge.second->ComputeResidual();
                tempChi += edge.second->Chi2();                     // Chi2() return residual_.transpose() * information_ * residual_
            }
            
            // ����rho (see "L3BundleAdjustment.pdf" eq 10)
            double rho = (currentChi_ - tempChi) / scale;

            // Nielsen ���� (see "L3BundleAdjustment.pdf" eq 13 �� "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" P4 4.1.1 ����3)
            if (rho > 0 && isfinite(tempChi))                       // last step was good, ������½�
            {
                // ���loss�½�
                double alpha = 1. - pow((2 * rho - 1), 3);
                alpha = std::min(alpha, 2. / 3.);                   // ��һ���� "L3BundleAdjustment.pdf" �� "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" ��û�У�Ŀ����Ϊ�˿���alpha����̫��
                double scaleFactor = (std::max)(1. / 3., alpha);
                currentLambda_ *= scaleFactor;
                ni_ = 2;
                currentChi_ = tempChi;                              // ������ʧ����
                return true;
            }
            else {
                // ���loss���
                currentLambda_ *= ni_;
                ni_ *= 2;
                return false;
            }
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
