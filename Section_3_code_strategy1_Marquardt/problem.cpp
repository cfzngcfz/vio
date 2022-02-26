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


        bool Problem::Solve(int iterations) {

            if (edges_.size() == 0 || verticies_.size() == 0) {
                std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
                return false;
            }

            TicToc t_solve;
            // 统计待优化变量的维数，为构建 H 矩阵做准备
            SetOrdering();
            // 遍历edge, 构建 H = J^T * J 矩阵
            MakeHessian();
            // LM 初始化
            ComputeLambdaInitLM(); // 计算阻尼因子mu的初始值
            // LM 算法迭代求解
            bool stop = false;
            int iter = 0;
            while (!stop && (iter < iterations)) {
                std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
                // currentChi_: 当前的损失函数值; currentLambda_: 当前的阻尼因子
                
                bool oneStepSuccess = false;
                int false_cnt = 0;
                while (!oneStepSuccess)  // 不断尝试 Lambda, 直到成功迭代一步  更新阻尼因子？
                {
                    // setLambda
                    AddLambdatoHessianLM(); // 对角线加上当前的阻尼因子,为了求解\Delat x_lm
                    // 第四步，解线性方程 H X = B
                    SolveLinearSystem();     // 求解 ( J_f^T * J_f + mu*I ) * \Delat x_lm = - J_f^T * f(x)
                    RemoveLambdaHessianLM(); // 对角线减去当前的阻尼因子，因为每次迭代中的阻尼因子是变化的

                    // 优化退出条件1： delta_x_ 很小则退出
                    if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10) {
                        stop = true;
                        break;
                    }

                    // 更新状态量 X = X+ delta_x
                    UpdateStates();
                    // 判断当前步是否可行以及 LM 的 lambda 怎么更新
                    oneStepSuccess = IsGoodStepInLM(); // 判断损失函数是否下降
                    // 后续处理，
                    if (oneStepSuccess) {
                        // 如果损失函数下降
      
                        // 在新线性化点 构建 hessian
                        MakeHessian();
                        //// TODO:: 这个判断条件可以丢掉，条件 b_max <= 1e-12 很难达到，这里的阈值条件不应该用绝对值，而是相对值
                        //double b_max = 0.0;
                        //for (int i = 0; i < b_.size(); ++i) {
                        //    b_max = max(fabs(b_(i)), b_max);
                        //}
                        //// 优化退出条件2： 如果残差 b_max 已经很小了，那就退出
                        //stop = (b_max <= 1e-12);
                        false_cnt = 0;
                    }
                    else {
                        // 如果损失函数没有下降
                        false_cnt++;
                        RollbackStates();   // 误差没下降，回滚，即 UpdateStates() 的逆操作
                    }
                }
                iter++;

                // 优化退出条件3： currentChi_ 跟第一次的chi2相比，下降了 1e6 倍则退出
                if (sqrt(currentChi_) <= stopThresholdLM_)  // stopThresholdLM_ defined in Problem::ComputeLambdaInitLM
                    stop = true;
            }
            std::cout << "---------" << std::endl;
            std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
            std::cout << "  makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
            return true;
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
            MatXX H(MatXX::Zero(size, size));
            VecX b(VecX::Zero(size));

            // TODO:: accelate, accelate, accelate
        //#ifdef USE_OPENMP
        //#pragma omp parallel for
        //#endif

            // 遍历每个残差，并计算他们的雅克比，得到最后的 H = J^T * J
            for (auto& edge : edges_) {

                edge.second->ComputeResidual();
                edge.second->ComputeJacobians();

                auto jacobians = edge.second->Jacobians();
                auto verticies = edge.second->Verticies();
                assert(jacobians.size() == verticies.size());
                for (size_t i = 0; i < verticies.size(); ++i) {
                    auto v_i = verticies[i];
                    if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

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
                        // 所有的信息矩阵叠加起来
                        H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                        if (j != i) {
                            // 对称的下三角
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
                ulong idx = vertex.second->OrderingId();   // 因为verticies_ 是 HashVertex, vertex.second 指的是verticies_的第二项(value), 即待优化的参数
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
                // Hessian_(i, i) += currentLambda_;
                Hessian_(i, i) += currentLambda_ * Hessian_(i, i);          // Marquardt 策略 修改1-4
                // use eq’n (13) for hlm  (see "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" P4)
                // 用 [J^T*W*J + 阻尼因子 * diag(J^T*W*J)] 更新 Hessian_ (see "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" P3 eq 13)
            }
        }

        void Problem::RemoveLambdaHessianLM() {
            ulong size = Hessian_.cols();
            assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
            // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
            for (ulong i = 0; i < size; ++i) {
                // Hessian_(i, i) -= currentLambda_;
                //Hessian_(i, i) -= currentLambda_ * Hessian_(i, i);          // 这样写错误
                Hessian_(i, i) /= (1.0 + currentLambda_);          // Marquardt 策略 修改2-4
            }
        }

        bool Problem::IsGoodStepInLM() {
            // 判断梯度是否下降

            // 计算rho的分母
            double scale = 0;
            // scale = delta_x_.transpose() * (currentLambda_ * delta_x_ + b_); // \Delta x_lm^T * (阻尼因子 * \Delta x_lm^T + b)

            // Marquardt 策略 修改3-4
            Eigen::Matrix3d diag = Eigen::Matrix3d::Zero();
            for (int ii = 0; ii < diag.rows(); ii++)
            {
                diag(ii, ii) = Hessian_(ii, ii);
            }
            scale = 0.5*delta_x_.transpose() * (currentLambda_ * diag * delta_x_ + b_);      // 需要推导确认前面是否*0.5！！！
            // (see "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" P3 eq 16)

            scale += 1e-3;    // make sure it's non-zero :)

            // recompute residuals after update state, 即 loss( x + \Delta x_{lm} )
            double tempChi = 0.0;                                   // 记录 x + \Delta x_{lm} 对应的损失函数/残差平方和
            for (auto edge : edges_) {
                edge.second->ComputeResidual();
                tempChi += edge.second->Chi2();                     // Chi2() return residual_.transpose() * information_ * residual_
            }
            
            // 计算rho (see "L3BundleAdjustment.pdf" eq 10)
            double rho = (currentChi_ - tempChi) / scale;

            // Marquardt 策略 (see "L3BundleAdjustment.pdf" eq 13 和 "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" P4 4.1.1 策略1)
            // Marquardt 策略 修改4-4
            if (rho > 0 && isfinite(tempChi))
            {
                currentLambda_ = (std::max)(currentLambda_ / 9., 1e-7);
                currentChi_ = tempChi;                              // 更新损失函数
                return true;
            }
            else
            {
                currentLambda_ = (std::min)(currentLambda_ * 9., 1e7); // 论文中是*11，改为*7-9
                return false;
            }

            //// Nielsen 策略 (see "L3BundleAdjustment.pdf" eq 13 和 "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" P4 4.1.1 策略3)
            //if (rho > 0 && isfinite(tempChi))                       // last step was good, 误差在下降
            //{
            //    // 如果loss下降
            //    double alpha = 1. - pow((2 * rho - 1), 3);
            //    alpha = std::min(alpha, 2. / 3.);                   // 这一步在 "L3BundleAdjustment.pdf" 和 "The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems.pdf" 都没有，目的是为了控制alpha不能太大
            //    double scaleFactor = (std::max)(1. / 3., alpha);
            //    currentLambda_ *= scaleFactor;
            //    ni_ = 2;
            //    currentChi_ = tempChi;                              // 更新损失函数
            //    return true;
            //}
            //else {
            //    // 如果loss变大
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
