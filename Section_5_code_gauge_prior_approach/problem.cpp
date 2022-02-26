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

// define the format you want, you only need one instance of this...
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void writeToCSVfile(std::string name, Eigen::MatrixXd matrix) {
    std::ofstream f(name.c_str());
    f << matrix.format(CSVFormat);
}

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

            if (problemType_ == ProblemType::SLAM_PROBLEM) {
                if (IsPoseVertex(vertex)) {
                    ResizePoseHessiansWhenAddingPose(vertex);
                }
            }
            return true;
        }

        void Problem::AddOrderingSLAM(std::shared_ptr<myslam::backend::Vertex> v) {
            if (IsPoseVertex(v)) {
                v->SetOrderingId(ordering_poses_);
                idx_pose_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
                ordering_poses_ += v->LocalDimension();
            }
            else if (IsLandmarkVertex(v)) {
                v->SetOrderingId(ordering_landmarks_);
                ordering_landmarks_ += v->LocalDimension();
                idx_landmark_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
            }
        }

        void Problem::ResizePoseHessiansWhenAddingPose(shared_ptr<Vertex> v) {

            int size = H_prior_.rows() + v->LocalDimension();
            H_prior_.conservativeResize(size, size);
            b_prior_.conservativeResize(size);

            b_prior_.tail(v->LocalDimension()).setZero();
            H_prior_.rightCols(v->LocalDimension()).setZero();
            H_prior_.bottomRows(v->LocalDimension()).setZero();

        }

        bool Problem::IsPoseVertex(std::shared_ptr<myslam::backend::Vertex> v) {
            string type = v->TypeInfo();
            return type == string("VertexPose");
        }

        bool Problem::IsLandmarkVertex(std::shared_ptr<myslam::backend::Vertex> v) {
            string type = v->TypeInfo();
            return type == string("VertexPointXYZ") ||
                type == string("VertexInverseDepth");
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

        vector<shared_ptr<Edge>> Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex) {
            vector<shared_ptr<Edge>> edges;
            auto range = vertexToEdge_.equal_range(vertex->Id());
            for (auto iter = range.first; iter != range.second; ++iter) {

                // �������edge����Ҫ���ڣ��������Ѿ���remove��
                if (edges_.find(iter->second->Id()) == edges_.end())
                    continue;

                edges.emplace_back(iter->second);
            }
            return edges;
        }

        bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex) {
            //check if the vertex is in map_verticies_
            if (verticies_.find(vertex->Id()) == verticies_.end()) {
                // LOG(WARNING) << "The vertex " << vertex->Id() << " is not in the problem!" << endl;
                return false;
            }

            // ����Ҫ remove �ö����Ӧ�� edge.
            vector<shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);
            for (size_t i = 0; i < remove_edges.size(); i++) {
                RemoveEdge(remove_edges[i]);
            }

            if (IsPoseVertex(vertex))
                idx_pose_vertices_.erase(vertex->Id());
            else
                idx_landmark_vertices_.erase(vertex->Id());

            vertex->SetOrderingId(-1);      // used to debug
            verticies_.erase(vertex->Id());
            vertexToEdge_.erase(vertex->Id());

            return true;
        }

        bool Problem::RemoveEdge(std::shared_ptr<Edge> edge) {
            //check if the edge is in map_edges_
            if (edges_.find(edge->Id()) == edges_.end()) {
                // LOG(WARNING) << "The edge " << edge->Id() << " is not in the problem!" << endl;
                return false;
            }

            edges_.erase(edge->Id());
            return true;
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
            // LM ��ʼ��
            ComputeLambdaInitLM();
            // LM �㷨�������
            bool stop = false;
            int iter = 0;
            while (!stop && (iter < iterations)) {
                std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
                bool oneStepSuccess = false;
                int false_cnt = 0;
                while (!oneStepSuccess)  // ���ϳ��� Lambda, ֱ���ɹ�����һ��
                {
                    // setLambda
        //            AddLambdatoHessianLM();
                    // ���Ĳ��������Է���
                    SolveLinearSystem();
                    //
        //            RemoveLambdaHessianLM();

                    // �Ż��˳�����1�� delta_x_ ��С���˳�
                    if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10) {
                        stop = true;
                        break;
                    }

                    // ����״̬��
                    UpdateStates();
                    // �жϵ�ǰ���Ƿ�����Լ� LM �� lambda ��ô����
                    oneStepSuccess = IsGoodStepInLM();
                    // ��������
                    if (oneStepSuccess) {
                        // �������Ի��� ���� hessian
                        MakeHessian();
                        // TODO:: ����ж��������Զ��������� b_max <= 1e-12 ���Ѵﵽ���������ֵ������Ӧ���þ���ֵ���������ֵ
        //                double b_max = 0.0;
        //                for (int i = 0; i < b_.size(); ++i) {
        //                    b_max = max(fabs(b_(i)), b_max);
        //                }
        //                // �Ż��˳�����2�� ����в� b_max �Ѿ���С�ˣ��Ǿ��˳�
        //                stop = (b_max <= 1e-12);
                        false_cnt = 0;
                    }
                    else {
                        false_cnt++;
                        RollbackStates();   // ���û�½����ع�
                    }
                }
                iter++;

                // �Ż��˳�����3�� currentChi_ ����һ�ε�chi2��ȣ��½��� 1e6 �����˳�
                if (sqrt(currentChi_) <= stopThresholdLM_)
                    stop = true;
            }
            std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
            std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
            return true;
        }

        void Problem::SetOrdering() {

            // ÿ�����¼���
            ordering_poses_ = 0;
            ordering_generic_ = 0;
            ordering_landmarks_ = 0;
            int debug = 0;

            cout << "\n===== SetOrdering() in Solve =====" << endl;
            for (auto vertex : verticies_) {
                cout << vertex.second->Id() << ", TypeInfo = " << vertex.second->TypeInfo() << ", OrderingId before SetOrdering and AddOrderingSLAM = " << vertex.second->OrderingId() << endl;
                // HashVertex verticies_; defined in Problem.h
                // verticies_.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex)); defined in Problem::AddVertex
            }
            
            for (auto vertex : verticies_) {
                ordering_generic_ += vertex.second->LocalDimension();  // ���е��Ż�������ά��

                if (IsPoseVertex(vertex.second)) {
                    debug += vertex.second->LocalDimension();
                }

                if (problemType_ == ProblemType::SLAM_PROBLEM)    // ����� slam ���⣬��Ҫ�ֱ�ͳ�� pose �� landmark ��ά�������������ǽ�������
                {
                    AddOrderingSLAM(vertex.second);
                }
                
                /*if (IsPoseVertex(vertex.second)) {
                    std::cout << vertex.second->Id() << " order: " << vertex.second->OrderingId() << std::endl;
                }*/
            }

            for (auto vertex : verticies_) {
                cout << vertex.second->Id() << ", TypeInfo = " << vertex.second->TypeInfo() << ", OrderingId after AddOrderingSLAM = " << vertex.second->OrderingId() << endl;
            }

            cout << "\nordering_generic = " << ordering_generic_ << ", ordering_poses = " << ordering_poses_ << ", ordering_landmarks = " << ordering_landmarks_ << "\n" << endl;
            // std::cout << "\n ordered_landmark_vertices_ size : " << idx_landmark_vertices_.size() << std::endl;
            if (problemType_ == ProblemType::SLAM_PROBLEM) {
                // ����Ҫ�� landmark �� ordering ���� pose ���������ͱ����� landmark �ں�,�� pose ��ǰ
                ulong all_pose_dimension = ordering_poses_;
                for (auto landmarkVertex : idx_landmark_vertices_) {
                    //cout << landmarkVertex.second->OrderingId() << ", ";
                    landmarkVertex.second->SetOrderingId( landmarkVertex.second->OrderingId() + all_pose_dimension ); // ����landmarkVertex/����ȶ���� OrderingId()������PoseVertex�� OrderingId()����
                    //cout << landmarkVertex.second->OrderingId() << endl;
                }
            }

            for (auto vertex : verticies_) {
                cout << vertex.second->Id() << ", TypeInfo = " << vertex.second->TypeInfo() << ", OrderingId after SetOrdering = " << vertex.second->OrderingId() << endl;
            }
            //    CHECK_EQ(CheckOrdering(), true);
        }


        bool Problem::CheckOrdering() {
            if (problemType_ == ProblemType::SLAM_PROBLEM) {
                int current_ordering = 0;
                for (auto v : idx_pose_vertices_) {
                    assert(v.second->OrderingId() == current_ordering);
                    current_ordering += v.second->LocalDimension();
                }

                for (auto v : idx_landmark_vertices_) {
                    assert(v.second->OrderingId() == current_ordering);
                    current_ordering += v.second->LocalDimension();
                }
            }
            return true;
        }

        void Problem::MakeHessian() {
            TicToc t_h;
            // ֱ�ӹ����� H ����
            ulong size = ordering_generic_;
            MatXX H(MatXX::Zero(size, size));
            VecX b(VecX::Zero(size));

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
                        // TODO:: home work. ��� H index ����д.
                        H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                        if (j != i) {
                            // �ԳƵ�������
                            // TODO:: home work. ��� H index ����д.
                            H.block(index_j,index_i, dim_j, dim_i).noalias() += hessian.transpose();
                        }
                    }
                    b.segment(index_i, dim_i).noalias() -= JtW * edge.second->Residual();
                }

            }
            Hessian_ = H;
            b_ = b;
            t_hessian_cost_ += t_h.toc();


            //    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
            //    std::cout << svd.singularValues() <<std::endl;

            if (err_prior_.rows() > 0) {
                b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);   // update the error_prior
            }
            Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_;
            b_.head(ordering_poses_) += b_prior_;

            delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;

        }

        /*
         * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
         */
        void Problem::SolveLinearSystem() {

            if (problemType_ == ProblemType::GENERIC_PROBLEM) {

                // �� SLAM ����ֱ�����
                // PCG solver
                MatXX H = Hessian_;
                for (ulong i = 0; i < Hessian_.cols(); ++i) {
                    H(i, i) += currentLambda_;
                }
                //        delta_x_ = PCGSolver(H, b_, H.rows() * 2);
                delta_x_ = Hessian_.inverse() * b_;

            }
            else {

                // SLAM �������������ļ��㷽ʽ
                // step1: schur marginalization --> Hpp, bpp
                int reserve_size = ordering_poses_;
                int marg_size = ordering_landmarks_;

                // TODO:: home work. ��ɾ����ȡֵ��Hmm��Hpm��Hmp��bpp��bmm
                MatXX Hmm = Hessian_.block(reserve_size,reserve_size, marg_size, marg_size);
                MatXX Hpm = Hessian_.block(0,reserve_size, reserve_size, marg_size);
                MatXX Hmp = Hessian_.block(reserve_size,0, marg_size, reserve_size);
                VecX bpp = b_.segment(0,reserve_size);
                VecX bmm = b_.segment(reserve_size,marg_size);

                // Hmm �ǶԽ��߾��������������ֱ��Ϊ�Խ��߿�ֱ����棬���������ȣ��Խ��߿�Ϊ1ά�ģ���ֱ��Ϊ�Խ��ߵĵ�����������Լ���
                MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
                for (auto landmarkVertex : idx_landmark_vertices_) {
                    int idx = landmarkVertex.second->OrderingId() - reserve_size;
                    int size = landmarkVertex.second->LocalDimension();
                    Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
                }

                // TODO:: home work. �������� Hpp, bpp ����
                MatXX tempH = Hpm * Hmm_inv;
                H_pp_schur_ = Hessian_.block(0,0,reserve_size,reserve_size) - tempH * Hmp;
                b_pp_schur_ = bpp - tempH * bmm;

                // step2: solve Hpp * delta_x = bpp
                VecX delta_x_pp(VecX::Zero(reserve_size));
                // PCG Solver
                for (ulong i = 0; i < ordering_poses_; ++i) {
                    H_pp_schur_(i, i) += currentLambda_;
                }
                int n = H_pp_schur_.rows() * 2;                       // ��������
                delta_x_pp = PCGSolver(H_pp_schur_, b_pp_schur_, n);  // ������С��ģ���⣬�� pcg �������
                delta_x_.head(reserve_size) = delta_x_pp;
                //        std::cout << delta_x_pp.transpose() << std::endl;

                // TODO:: home work. step3: solve landmark
                VecX delta_x_ll(marg_size);
                delta_x_ll = Hmm_inv*(bmm - Hmp * delta_x_pp);
                delta_x_.tail(marg_size) = delta_x_ll;

            }

        }

        void Problem::UpdateStates() {
            for (auto vertex : verticies_) {
                ulong idx = vertex.second->OrderingId();
                ulong dim = vertex.second->LocalDimension();
                VecX delta = delta_x_.segment(idx, dim);
                vertex.second->Plus(delta);
            }
            if (err_prior_.rows() > 0) {
                b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);   // update the error_prior
                err_prior_ = Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 6);
            }

        }

        void Problem::RollbackStates() {
            for (auto vertex : verticies_) {
                ulong idx = vertex.second->OrderingId();
                ulong dim = vertex.second->LocalDimension();
                VecX delta = delta_x_.segment(idx, dim);
                vertex.second->Plus(-delta);
            }
            if (err_prior_.rows() > 0) {
                b_prior_ += H_prior_ * delta_x_.head(ordering_poses_);   // update the error_prior
                err_prior_ = Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 6);
            }
        }

        /// LM
        void Problem::ComputeLambdaInitLM() {
            ni_ = 2.;
            currentLambda_ = -1.;
            currentChi_ = 0.0;
            // TODO:: robust cost chi2
            for (auto edge : edges_) {
                currentChi_ += edge.second->Chi2();
            }
            if (err_prior_.rows() > 0)      // marg prior residual
                currentChi_ += err_prior_.norm();

            stopThresholdLM_ = 1e-6 * currentChi_;          // ��������Ϊ ����½� 1e-6 ��

            double maxDiagonal = 0;
            ulong size = Hessian_.cols();
            assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
            for (ulong i = 0; i < size; ++i) {
                maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
            }
            double tau = 1e-5;
            currentLambda_ = tau * maxDiagonal;
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
            double scale = 0;
            scale = delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
            scale += 1e-3;    // make sure it's non-zero :)

            // recompute residuals after update state
            // TODO:: get robustChi2() instead of Chi2()
            double tempChi = 0.0;
            for (auto edge : edges_) {
                edge.second->ComputeResidual();
                tempChi += edge.second->Chi2();
            }
            if (err_prior_.size() > 0)
                tempChi += err_prior_.norm();

            double rho = (currentChi_ - tempChi) / scale;
            if (rho > 0 && isfinite(tempChi))   // last step was good, ������½�
            {
                double alpha = 1. - pow((2 * rho - 1), 3);
                alpha = std::min(alpha, 2. / 3.);
                double scaleFactor = (std::max)(1. / 3., alpha);
                currentLambda_ *= scaleFactor;
                ni_ = 2;
                currentChi_ = tempChi;
                return true;
            }
            else {
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

        /*
         *  marg ���к� frame ������ edge: imu factor, projection factor�� prior factor
         *
         */
        bool Problem::Marginalize(const std::shared_ptr<Vertex> frameVertex) {

            return true;
        }


        void Problem::TestMarginalize() {

            // Add marg test
            int idx = 1;            // marg �м��Ǹ�����
            int dim = 1;            // marg ������ά��
            int reserve_size = 3;   // �ܹ�������ά��
            double delta1 = 0.1 * 0.1;
            double delta2 = 0.2 * 0.2;
            double delta3 = 0.3 * 0.3;

            int cols = 3;
            MatXX H_marg(MatXX::Zero(cols, cols));
            H_marg << 1. / delta1, -1. / delta1, 0,
                -1. / delta1, 1. / delta1 + 1. / delta2 + 1. / delta3, -1. / delta3,
                0., -1. / delta3, 1 / delta3;
            std::cout << "---------- TEST Marg: before marg------------" << std::endl;
            std::cout << H_marg << std::endl;

            // TODO:: home work. �������ƶ������½�
            /// ׼�������� move the marg pose to the Hmm bottown right
            // �� row i �ƶ�����������
            Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);
            Eigen::MatrixXd temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
            H_marg.block(idx,0,reserve_size - idx - dim,reserve_size) = temp_botRows;
            H_marg.block(reserve_size - dim,0,dim,reserve_size) = temp_rows;

            // �� col i �ƶ��������ұ�
            Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim);
            Eigen::MatrixXd temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
            H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols;
            H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;

            std::cout << "---------- TEST Marg: �������ƶ������½�------------" << std::endl;
            std::cout << H_marg << std::endl;

            /// ��ʼ marg �� schur
            double eps = 1e-8;
            int m2 = dim;
            int n2 = reserve_size - dim;   // ʣ�������ά��
            Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
            Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
                (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
                saes.eigenvectors().transpose();

            // TODO:: home work. ������������
            Eigen::MatrixXd Arm = H_marg.block(0, n2, n2, m2);
            Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);
            Eigen::MatrixXd Arr = H_marg.block(0, 0, n2, n2);

            Eigen::MatrixXd tempB = Arm * Amm_inv;
            Eigen::MatrixXd H_prior = Arr - tempB * Amr;

            std::cout << "---------- TEST Marg: after marg------------" << std::endl;
            std::cout << H_prior << std::endl;
        }

    }
}