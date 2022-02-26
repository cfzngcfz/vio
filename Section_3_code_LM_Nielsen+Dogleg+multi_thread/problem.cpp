#include <iostream>
#include <fstream>
#include <Eigen/Dense>
//#include <glog/logging.h>
#include "problem.h"
#include "tic_toc.h"
#include <thread>
#include <mutex> // 互斥锁

#ifdef USE_OPENMP

#include <omp.h>

#endif

using namespace std;


// 控制使用LM 或者Dogleg
const int optimization_method = 1; // 0: LM,   1: Dogleg
// 控制是否使用加速, 以及加速方式
const int acceleration_method = 1; // 0: Normal,non-acc,  1: multi-threads acc
const int num_thread = 4;

// 定义最小二乘问题

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
            // 统计优化变量的维数，为构建 H 矩阵做准备
            SetOrdering();
            // 遍历edge, 构建 H 矩阵
            MakeHessian();

            bool stop = false;
            int iter = 0;
            double last_chi_ = 1e20;

            if (optimization_method == 0) { // LM算法
              // LM 初始化
                ComputeLambdaInitLM();
                // LM 算法迭代求解
                while (!stop && (iter < iterations)) {
                    std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
                    bool oneStepSuccess = false;
                    int false_cnt = 0;
                    while (!oneStepSuccess && false_cnt < 10)  // 不断尝试 Lambda, 直到成功迭代一步
                    {
                        AddLambdatoHessianLM(); // 对角线加上当前的阻尼因子
                        SolveLinearSystem();
                        RemoveLambdaHessianLM(); // 对角线减去当前的阻尼因子，因为每次迭代中的阻尼因子是变化的

                        // 更新状态量
                        UpdateStates();
                        // 判断当前步是否可行以及 LM 的 lambda 怎么更新, chi2 也计算一下
                        oneStepSuccess = IsGoodStepInLM();
                        // 后续处理，
                        if (oneStepSuccess) {
                            // 在新线性化点 构建 hessian
                            MakeHessian();
                            false_cnt = 0;
                        }
                        else {
                            false_cnt++;
                            RollbackStates();  // 误差没下降，回滚
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

              // 仅初始化损失函数
                currentChi_ = 0.0;
                for (auto edge : edges_) {
                    // 在MakeHessian()中已经计算了edge.second->ComputeResidual()
                    currentChi_ += edge.second->Chi2();
                }
                if (err_prior_.rows() > 0)
                    currentChi_ += err_prior_.norm();

                radius_ = 1e4;  // 初始信赖域

                while (!stop && (iter < iterations)) {
                    std::cout << "\niter: " << iter << " , currentChi= " << currentChi_ << " , radius= " << radius_ << std::endl;
                    iter++;

                    bool oneStepSuccess = false;
                    int false_cnt = 0;
                    while (!oneStepSuccess && false_cnt < 10)  // 不断尝试 Lambda, 直到成功迭代一步
                    {
                        // step 2.1. 最速下降法 计算 h_sd_
                        double numerator = b_.transpose() * b_;
                        double denominator = b_.transpose() * Hessian_ * b_;
                        double alpha_ = numerator / denominator;
                        h_sd_ = alpha_ * b_;

                        // step 2.2. 高斯牛顿法 计算 h_gn_
                        // To Do: 此处Hessian_比较大, 直接求逆很耗时, 可采用 Gauss-Newton法求解
                        //h_gn_ = Hessian_.inverse() * b_;
                        h_gn_ = Hessian_.ldlt().solve(b_);

                        // 3.计算h_dl 步长
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
                        // 后续处理，
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

            // 计算损失函数
            double tempChi = 0.0;
            for (auto edge : edges_) {
                edge.second->ComputeResidual();
                tempChi += edge.second->Chi2();
            }
            if (err_prior_.size() > 0)
                //tempChi += err_prior_.norm();
                tempChi += err_prior_.norm();


            // 计算rho的分母
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

            // 计算rho
            double rho_ = (currentChi_ - tempChi) / scale;


            if (rho_ > 0.75 && isfinite(tempChi)) {
                radius_ = std::max(radius_, 3 * delta_x_.norm());
            }
            else if (rho_ < 0.25) {
                radius_ = std::max(radius_ / 4, 1e-7);
                // radius_ = 0.5 * radius_; // 论文中
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

            // 每次重新计数
            ordering_poses_ = 0;
            ordering_generic_ = 0;
            ordering_landmarks_ = 0;

            // Note:: verticies_ 是 map 类型的, 顺序是按照 id 号排序的
            // 统计带估计的所有变量的总维度
            for (auto vertex : verticies_) {
                ordering_generic_ += vertex.second->LocalDimension();  // 所有的优化变量总维数
            }
        }

        void Problem::MakeHessian() {
            TicToc t_h;
            // 直接构造大的 H 矩阵
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
                    if (v_i->IsFixed()) continue;  // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

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

                        // 所有的信息矩阵叠加起来
                        if (acceleration_method == 1) {
                            m_H_.lock();
                        }
                        H_all_.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                        if (j != i) {
                            // 对称的下三角
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

                // 所有的参数 x 叠加一个增量  x_{k+1} = x_{k} + delta_x
                vertex.second->Plus(delta);
            }
        }

        void Problem::RollbackStates() {
            for (auto vertex : verticies_) {
                ulong idx = vertex.second->OrderingId();
                ulong dim = vertex.second->LocalDimension();
                VecX delta = delta_x_.segment(idx, dim);

                // 之前的增量加了后使得损失函数增加了，我们应该不要这次迭代结果，所以把之前加上的量减去。
                vertex.second->Plus(-delta);
            }
        }

        /// LM
        void Problem::ComputeLambdaInitLM() {
            ni_ = 2.;
            currentLambda_ = -1.;                                           // 记录阻尼因子
            currentChi_ = 0.0;                                              // 记录损失函数值
            // TODO:: robust cost chi2
            for (auto edge : edges_) {                                      // edges_ 记录每条边上的残差，即residual_
                currentChi_ += edge.second->Chi2();                         // Chi2() return residual_.transpose() * information_ * residual_
            }
            if (err_prior_.rows() > 0)
                currentChi_ += err_prior_.norm();

            stopThresholdLM_ = 1e-6 * currentChi_;                          // 迭代条件为 误差下降 1e-6 倍

            double maxDiagonal = 0;
            ulong size = Hessian_.cols();
            assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
            for (ulong i = 0; i < size; ++i) {
                maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal); // 取Hessian矩阵的主对角线元素的最大值
            }
            double tau = 1e-5;                                              // 自定义 阻尼因子与Hessian矩阵的主对角线元素的最大值 量纲之间的关系
            currentLambda_ = tau * maxDiagonal;                             // 初始的阻尼因子
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
            // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
            for (ulong i = 0; i < size; ++i) {
                Hessian_(i, i) -= currentLambda_;
            }
        }

        bool Problem::IsGoodStepInLM() {
            // 判断梯度是否下降

            // 计算rho的分母
            double scale = 0;
            scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_); // \Delta x_lm^T * (阻尼因子 * \Delta x_lm^T + b)
            scale += 1e-3;    // make sure it's non-zero :)

            // recompute residuals after update state, 即 loss( x + \Delta x_{lm} )
            double tempChi = 0.0;                                   // 记录 x + \Delta x_{lm} 对应的损失函数/残差平方和
            for (auto edge : edges_) {
                edge.second->ComputeResidual();
                tempChi += edge.second->Chi2();                     // Chi2() return residual_.transpose() * information_ * residual_
            }
            
            // 计算rho (see "L3BundleAdjustment.pdf" eq 10)
            double rho = (currentChi_ - tempChi) / scale;

            // Nielsen 策略 (see "L3BundleAdjustment.pdf" eq 13 和 "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" P4 4.1.1 策略3)
            if (rho > 0 && isfinite(tempChi))                       // last step was good, 误差在下降
            {
                // 如果loss下降
                double alpha = 1. - pow((2 * rho - 1), 3);
                alpha = std::min(alpha, 2. / 3.);                   // 这一步在 "L3BundleAdjustment.pdf" 和 "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" 都没有，目的是为了控制alpha不能太大
                double scaleFactor = (std::max)(1. / 3., alpha);
                currentLambda_ *= scaleFactor;
                ni_ = 2;
                currentChi_ = tempChi;                              // 更新损失函数
                return true;
            }
            else {
                // 如果loss变大
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
