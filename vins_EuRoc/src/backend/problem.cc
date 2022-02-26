#include "backend/problem.h"
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <mutex> // 互斥�?
#include "utility/tic_toc.h"

#ifdef USE_OPENMP

#include <omp.h>

#endif

using namespace std;

// 控制使用LM 或者Dogleg
const int optimization_method = 1; // 0: LM,   1: Dogleg
// 控制是否使用加�?, 以及加速方�?
const int acceleration_method = 0; // 0: Normal,non-acc,  1: multi-threads acc
const int num_thread = 1;

//// define the format you want, you only need one instance of this...
//const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
//
//void writeToCSVfile(std::string name, Eigen::MatrixXd matrix) {
//  std::ofstream f(name.c_str());
//  f << matrix.format(CSVFormat);
//}

namespace myslam {
namespace backend {
void Problem::LogoutVectorSize() {
  // LOG(INFO) <<
  //           "1 problem::LogoutVectorSize verticies_:" << verticies_.size() <<
  //           " edges:" << edges_.size();
}

Problem::Problem(ProblemType problemType) : problemType_(problemType) {
  LogoutVectorSize();
  verticies_marg_.clear();
}

Problem::~Problem() {
  //    std::cout << "Problem IS Deleted"<<std::endl;
  global_vertex_id = 0;
}

bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) {
  if (verticies_.find(vertex->Id()) != verticies_.end()) {
    // LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
    return false;
  } else {
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
  } else if (IsLandmarkVertex(v)) {
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
void Problem::ExtendHessiansPriorSize(int dim) {
  int size = H_prior_.rows() + dim;
  H_prior_.conservativeResize(size, size);
  b_prior_.conservativeResize(size);

  b_prior_.tail(dim).setZero();
  H_prior_.rightCols(dim).setZero();
  H_prior_.bottomRows(dim).setZero();
}

bool Problem::IsPoseVertex(std::shared_ptr<myslam::backend::Vertex> v) {
  string type = v->TypeInfo();
  return type == string("VertexPose") || type == string("VertexSpeedBias");
}

bool Problem::IsLandmarkVertex(std::shared_ptr<myslam::backend::Vertex> v) {
  string type = v->TypeInfo();
  return type == string("VertexPointXYZ") || type == string("VertexInverseDepth");
}

bool Problem::AddEdge(shared_ptr<Edge> edge) {
  if (edges_.find(edge->Id()) == edges_.end()) {
    edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
  } else {
    // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
    return false;
  }

  for (auto &vertex : edge->Verticies()) {
    vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
  }
  return true;
}

vector<shared_ptr<Edge>> Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex) {
  vector<shared_ptr<Edge>> edges;
  auto range = vertexToEdge_.equal_range(vertex->Id());
  for (auto iter = range.first; iter != range.second; ++iter) {
    // 并且这个edge还需要存在，而不是已经被remove�?
    if (edges_.find(iter->second->Id()) == edges_.end()) continue;

    edges.emplace_back(iter->second);
  }
  return edges;
}

bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex) {
  // check if the vertex is in map_verticies_
  if (verticies_.find(vertex->Id()) == verticies_.end()) {
    // LOG(WARNING) << "The vertex " << vertex->Id() << " is not in the problem!" << endl;
    return false;
  }

  // 这里�? remove 该顶点对应的 edge.
  vector<shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);
  for (size_t i = 0; i < remove_edges.size(); i++) {
    RemoveEdge(remove_edges[i]);
  }

  if (IsPoseVertex(vertex))
    idx_pose_vertices_.erase(vertex->Id());
  else
    idx_landmark_vertices_.erase(vertex->Id());

  vertex->SetOrderingId(-1);  // used to debug
  verticies_.erase(vertex->Id());
  vertexToEdge_.erase(vertex->Id());

  return true;
}

bool Problem::RemoveEdge(std::shared_ptr<Edge> edge) {
  // check if the edge is in map_edges_
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
    // 统计优化变量的维数，为构�? H 矩阵做准�?
    SetOrdering();
    // 遍历edge, 构建 H 矩阵
    MakeHessian();

    bool stop = false;
    int iter = 0;
    double last_chi_ = 1e20;

    if (optimization_method == 0) { // LM算法
      // LM 初始�?
        ComputeLambdaInitLM();
        // LM 算法迭代求解
        while (!stop && (iter < iterations)) {
            std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ << std::endl;
            bool oneStepSuccess = false;
            int false_cnt = 0;
            while (!oneStepSuccess && false_cnt < 10)  // 不断尝试 Lambda, 直到成功迭代一�?
            {
                SolveLinearSystem();
                // 更新状态量
                UpdateStates();
                // 判断当前步是否可行以�? LM �? lambda 怎么更新, chi2 也计算一�?
                oneStepSuccess = IsGoodStepInLM();
                // 后续处理�?
                if (oneStepSuccess) {
                    // 在新线性化�? 构建 hessian
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
            currentChi_ += edge.second->RobustChi2();
        }
        if (err_prior_.rows() > 0)
            currentChi_ += err_prior_.squaredNorm();
        currentChi_ *= 0.5;

        radius_ = 1e4;  // 初始信赖�?

        while (!stop && (iter < iterations)) {
            std::cout << "iter: " << iter << " , currentChi= " << currentChi_ << " , radius= " << radius_ << std::endl;
            iter++;

            bool oneStepSuccess = false;
            int false_cnt = 0;
            while (!oneStepSuccess && false_cnt < 10)  // 不断尝试 Lambda, 直到成功迭代一�?
            {
                // step 2.1. 最速下降法 计算 h_sd_
                double numerator = b_.transpose() * b_;
                double denominator = b_.transpose() * Hessian_ * b_;
                double alpha_ = numerator / denominator;
                h_sd_ = alpha_ * b_; //�����h_sd_��delta x�������½�����

                // step 2.2. 高斯牛顿�? 计算 h_gn_
                // To Do: 此处Hessian_比较�?, 直接求逆很耗时, 可采�? Gauss-Newton法求�?
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
                // 后续处理�?
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

    time_solve_ = t_solve.toc();
    time_makeHessian_ = t_hessian_cost_;

    // 数据保存
    std::string filename = "./time_output.txt";
    std::ofstream fs;
    fs.open(filename, std::ios::app); 
    if (!fs.is_open()) {
        cerr << "Failed to open image file! " << filename << endl;
    }
    fs << time_solve_ << " " << time_makeHessian_ << std::endl;
    fs.close();


    t_hessian_cost_ = 0.;
    return true;
}

bool Problem::IsGoodStepInDogleg() {

    // ������ʧ����
    double tempChi = 0.0;
    for (auto edge : edges_) {
        edge.second->ComputeResidual();
        tempChi += edge.second->RobustChi2();
    }
    if (err_prior_.size() > 0)
        tempChi += err_prior_.squaredNorm();
    tempChi *= 0.5;          // 1/2 * err^2


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
        // radius_ = 0.5 * radius_;
    }


    if (rho_ > 0 && isfinite(tempChi)) {
        currentChi_ = tempChi;
        return true;
    }
    else {
        return false;
    }
}

bool Problem::SolveGenericProblem(int iterations) { return true; }

void Problem::SetOrdering() {
  // 每次重新计数
  ordering_poses_ = 0;
  ordering_generic_ = 0;
  ordering_landmarks_ = 0;

  // Note:: verticies_ �? map 类型�?, 顺序是按�? id 号排序的
  for (auto vertex : verticies_) {
    ordering_generic_ += vertex.second->LocalDimension();  // 所有的优化变量总维�?

    if (problemType_ == ProblemType::SLAM_PROBLEM)  // 如果�? slam 问题，还要分别统�? pose �? landmark 的维数，后面会对他们进行排序
    {
      AddOrderingSLAM(vertex.second);
    }
  }

  if (problemType_ == ProblemType::SLAM_PROBLEM) {
    // 这里要把 landmark �? ordering 加上 pose 的数量，就保持了 landmark 在后,�? pose 在前
    ulong all_pose_dimension = ordering_poses_;
    for (auto landmarkVertex : idx_landmark_vertices_) {
      landmarkVertex.second->SetOrderingId(landmarkVertex.second->OrderingId() + all_pose_dimension);
    }
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
            vec_id[edge.first%num_thread].emplace_back(edge.first);
        }
        //for (int ii=0;ii<num_thread;ii++) {
        //  cout << "size of vec_id[" << ii << "] = " << vec_id[ii].size() << endl;
        //}
        for (int ii=0;ii<num_thread;ii++) {
            std::thread sub_thread = std::thread(&Problem::edge_thread, this, vec_id[ii]);
            sub_thread.join();
        }
    }

    Hessian_ = H_all_;
    b_ = b_all_;
    t_hessian_cost_ += t_h.toc();

    if (H_prior_.rows() > 0) {
        MatXX H_prior_tmp = H_prior_;
        VecX b_prior_tmp = b_prior_;

        /// �������� POSE ���㣬Ȼ��������Ӧ������ά��Ϊ 0 .  fix �����, SET PRIOR TO ZERO
        /// landmark û������
        for (auto vertex : verticies_) {
            if (IsPoseVertex(vertex.second) && vertex.second->IsFixed()) {
                int idx = vertex.second->OrderingId();
                int dim = vertex.second->LocalDimension();
                H_prior_tmp.block(idx, 0, dim, H_prior_tmp.cols()).setZero();
                H_prior_tmp.block(0, idx, H_prior_tmp.rows(), dim).setZero();
                b_prior_tmp.segment(idx, dim).setZero();
                //                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
            }
        }
        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        b_.head(ordering_poses_) += b_prior_tmp;
    }

    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;
}


void Problem::edge_thread(std::vector<ulong> vec_id) {
    for (ulong id : vec_id) {
        std::shared_ptr<Edge> edge_second = edges_[id];
        edge_second->ComputeResidual();
        edge_second->ComputeJacobians();

        // TODO:: robust cost
        auto jacobians = edge_second->Jacobians();
        auto verticies = edge_second->Verticies();
        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            if (v_i->IsFixed()) continue;  // Hessian �ﲻ��Ҫ����������Ϣ��Ҳ���������ſɱ�Ϊ 0

            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            // ³���˺������޸Ĳв����Ϣ�������û������ robust cost function���ͻ᷵��ԭ����
            double drho;
            MatXX robustInfo(edge_second->Information().rows(), edge_second->Information().cols());
            edge_second->RobustInfo(drho, robustInfo);

            MatXX JtW = jacobian_i.transpose() * robustInfo;
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
                    // �ԳƵ������ǧ�
                    H_all_.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();
                }
                if (acceleration_method == 1) {
                    m_H_.unlock();
                }
            }
            if (acceleration_method == 1) {
                m_b_.lock();
            }
            b_all_.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose() * edge_second->Information() * edge_second->Residual();
            if (acceleration_method == 1){
                m_b_.unlock();
            }
        }
    }
}

/*
 * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
 */
void Problem::SolveLinearSystem() {
  if (problemType_ == ProblemType::GENERIC_PROBLEM) {
    // PCG solver
    MatXX H = Hessian_;
    for (size_t i = 0; i < Hessian_.cols(); ++i) {
      H(i, i) += currentLambda_;
    }
    // delta_x_ = PCGSolver(H, b_, H.rows() * 2);
    delta_x_ = H.ldlt().solve(b_);

  } else {
    //        TicToc t_Hmminv;
    // step1: schur marginalization --> Hpp, bpp
    int reserve_size = ordering_poses_;
    int marg_size = ordering_landmarks_;
    MatXX Hmm = Hessian_.block(reserve_size, reserve_size, marg_size, marg_size);
    MatXX Hpm = Hessian_.block(0, reserve_size, reserve_size, marg_size);
    MatXX Hmp = Hessian_.block(reserve_size, 0, marg_size, reserve_size);
    VecX bpp = b_.segment(0, reserve_size);
    VecX bmm = b_.segment(reserve_size, marg_size);

    // Hmm 是对角线矩阵，它的求逆可以直接为对角线块分别求逆，如果是逆深度，对角线块�?1维的，则直接为对角线的倒数，这里可以加�?
    MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
    // TODO:: use openMP
    for (auto landmarkVertex : idx_landmark_vertices_) {
      int idx = landmarkVertex.second->OrderingId() - reserve_size;
      int size = landmarkVertex.second->LocalDimension();
      Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
    }

    MatXX tempH = Hpm * Hmm_inv;
    H_pp_schur_ = Hessian_.block(0, 0, ordering_poses_, ordering_poses_) - tempH * Hmp;
    b_pp_schur_ = bpp - tempH * bmm;

    // step2: solve Hpp * delta_x = bpp
    VecX delta_x_pp(VecX::Zero(reserve_size));

    for (ulong i = 0; i < ordering_poses_; ++i) {
      H_pp_schur_(i, i) += currentLambda_;  // LM Method
    }

    // TicToc t_linearsolver;
    delta_x_pp = H_pp_schur_.ldlt().solve(b_pp_schur_);  //  SVec.asDiagonal() * svd.matrixV() * Ub;
    delta_x_.head(reserve_size) = delta_x_pp;
    // std::cout << " Linear Solver Time Cost: " << t_linearsolver.toc() << std::endl;

    // step3: solve Hmm * delta_x = bmm - Hmp * delta_x_pp;
    VecX delta_x_ll(marg_size);
    delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
    delta_x_.tail(marg_size) = delta_x_ll;

    //        std::cout << "schur time cost: "<< t_Hmminv.toc()<<std::endl;
  }
}

void Problem::UpdateStates() {
  // update vertex
  for (auto vertex : verticies_) {
    vertex.second->BackUpParameters();  // 保存上次的估计�?

    ulong idx = vertex.second->OrderingId();
    ulong dim = vertex.second->LocalDimension();
    VecX delta = delta_x_.segment(idx, dim);
    vertex.second->Plus(delta);
  }

  // update prior
  if (err_prior_.rows() > 0) {
    // BACK UP b_prior_
    b_prior_backup_ = b_prior_;
    err_prior_backup_ = err_prior_;

    /// update with first order Taylor, b' = b + \frac{\delta b}{\delta x} * \delta x
    /// \delta x = Computes the linearized deviation from the references (linearization points)
    b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);  // update the error_prior
    err_prior_ = -Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 15);

    //        std::cout << "                : "<< b_prior_.norm()<<" " <<err_prior_.norm()<< std::endl;
    //        std::cout << "     delta_x_ ex: "<< delta_x_.head(6).norm() << std::endl;
  }
}

void Problem::RollbackStates() {
  // update vertex
  for (auto vertex : verticies_) {
    vertex.second->RollBackParameters();
  }

  // Roll back prior_
  if (err_prior_.rows() > 0) {
    b_prior_ = b_prior_backup_;
    err_prior_ = err_prior_backup_;
  }
}

/// LM
void Problem::ComputeLambdaInitLM() {
  ni_ = 2.;
  currentLambda_ = -1.;
  currentChi_ = 0.0;

  for (auto edge : edges_) {
    currentChi_ += edge.second->RobustChi2();
  }
  if (err_prior_.rows() > 0)
      currentChi_ += err_prior_.squaredNorm();
  currentChi_ *= 0.5;

  stopThresholdLM_ = 1e-10 * currentChi_;  // 迭代条件�? 误差下降 1e-6 �?

  double maxDiagonal = 0;
  ulong size = Hessian_.cols();
  assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
  for (ulong i = 0; i < size; ++i) {
    maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
  }

  maxDiagonal = std::min(5e10, maxDiagonal);
  double tau = 1e-5;  // 1e-5
  currentLambda_ = tau * maxDiagonal;
  //        std::cout << "currentLamba_: "<<maxDiagonal<<" "<<currentLambda_<<std::endl;
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
  // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋�?
  for (ulong i = 0; i < size; ++i) {
    Hessian_(i, i) -= currentLambda_;
  }
}

bool Problem::IsGoodStepInLM() {
  double scale = 0;
  //    scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
  //    scale += 1e-3;    // make sure it's non-zero :)
  scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
  scale += 1e-6;  // make sure it's non-zero :)

  // recompute residuals after update state
  double tempChi = 0.0;
  for (auto edge : edges_) {
    edge.second->ComputeResidual();
    tempChi += edge.second->RobustChi2();
  }
  if (err_prior_.size() > 0)
      tempChi += err_prior_.squaredNorm();
  tempChi *= 0.5;  // 1/2 * err^2

  double rho = (currentChi_ - tempChi) / scale;
  if (rho > 0 && isfinite(tempChi))  // last step was good, 误差在下�?
  {
    double alpha = 1. - pow((2 * rho - 1), 3);
    alpha = std::min(alpha, 2. / 3.);
    double scaleFactor = (std::max)(1. / 3., alpha);
    currentLambda_ *= scaleFactor;
    ni_ = 2;
    currentChi_ = tempChi;
    return true;
  } else {
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
VecX Problem::PCGSolver(const MatXX &A, const VecX &b, int maxIter = -1) {
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
 *  marg 所有和 frame 相连�? edge: imu factor, projection factor
 *  如果某个landmark和该frame相连，但是又不想加入marg, 那就把改edge先去�?
 *
 */
bool Problem::Marginalize(const std::vector<std::shared_ptr<Vertex>> margVertexs, int pose_dim) {
  SetOrdering();
  /// 找到需�? marg �? edge, margVertexs[0] is frame, its edge contained pre-intergration
  std::vector<shared_ptr<Edge>> marg_edges = GetConnectedEdges(margVertexs[0]);

  std::unordered_map<int, shared_ptr<Vertex>> margLandmark;
  // 构建 Hessian 的时�? pose 的顺序不变，landmark的顺序要重新设定
  int marg_landmark_size = 0;
  //    std::cout << "\n marg edge 1st id: "<< marg_edges.front()->Id() << " end id: "<<marg_edges.back()->Id()<<std::endl;
  for (size_t i = 0; i < marg_edges.size(); ++i) {
    //        std::cout << "marg edge id: "<< marg_edges[i]->Id() <<std::endl;
    auto verticies = marg_edges[i]->Verticies();
    for (auto iter : verticies) {
      if (IsLandmarkVertex(iter) && margLandmark.find(iter->Id()) == margLandmark.end()) {
        iter->SetOrderingId(pose_dim + marg_landmark_size);
        margLandmark.insert(make_pair(iter->Id(), iter));
        marg_landmark_size += iter->LocalDimension();
      }
    }
  }
  //    std::cout << "pose dim: " << pose_dim <<std::endl;
  int cols = pose_dim + marg_landmark_size;
  /// 构建误差 H 矩阵 H = H_marg + H_pp_prior
  MatXX H_marg(MatXX::Zero(cols, cols));
  VecX b_marg(VecX::Zero(cols));
  int ii = 0;
  for (auto edge : marg_edges) {
    edge->ComputeResidual();
    edge->ComputeJacobians();
    auto jacobians = edge->Jacobians();
    auto verticies = edge->Verticies();
    ii++;

    assert(jacobians.size() == verticies.size());
    for (size_t i = 0; i < verticies.size(); ++i) {
      auto v_i = verticies[i];
      auto jacobian_i = jacobians[i];
      ulong index_i = v_i->OrderingId();
      ulong dim_i = v_i->LocalDimension();

      double drho;
      MatXX robustInfo(edge->Information().rows(), edge->Information().cols());
      edge->RobustInfo(drho, robustInfo);

      for (size_t j = i; j < verticies.size(); ++j) {
        auto v_j = verticies[j];
        auto jacobian_j = jacobians[j];
        ulong index_j = v_j->OrderingId();
        ulong dim_j = v_j->LocalDimension();

        MatXX hessian = jacobian_i.transpose() * robustInfo * jacobian_j;

        assert(hessian.rows() == v_i->LocalDimension() && hessian.cols() == v_j->LocalDimension());
        // 所有的信息矩阵叠加起来
        H_marg.block(index_i, index_j, dim_i, dim_j) += hessian;
        if (j != i) {
          // 对称的下三角
          H_marg.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
        }
      }
      b_marg.segment(index_i, dim_i) -= drho * jacobian_i.transpose() * edge->Information() * edge->Residual();
    }
  }
  std::cout << "edge factor cnt: " << ii << std::endl;

  /// marg landmark
  int reserve_size = pose_dim;
  if (marg_landmark_size > 0) {
    int marg_size = marg_landmark_size;
    MatXX Hmm = H_marg.block(reserve_size, reserve_size, marg_size, marg_size);
    MatXX Hpm = H_marg.block(0, reserve_size, reserve_size, marg_size);
    MatXX Hmp = H_marg.block(reserve_size, 0, marg_size, reserve_size);
    VecX bpp = b_marg.segment(0, reserve_size);
    VecX bmm = b_marg.segment(reserve_size, marg_size);

    // Hmm 是对角线矩阵，它的求逆可以直接为对角线块分别求逆，如果是逆深度，对角线块�?1维的，则直接为对角线的倒数，这里可以加�?
    MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
    // TODO:: use openMP
    for (auto iter : margLandmark) {
      int idx = iter.second->OrderingId() - reserve_size;
      int size = iter.second->LocalDimension();
      Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
    }

    MatXX tempH = Hpm * Hmm_inv;
    MatXX Hpp = H_marg.block(0, 0, reserve_size, reserve_size) - tempH * Hmp;
    bpp = bpp - tempH * bmm;
    H_marg = Hpp;
    b_marg = bpp;
  }

  VecX b_prior_before = b_prior_;
  if (H_prior_.rows() > 0) {
    H_marg += H_prior_;
    b_marg += b_prior_;
  }

  /// marg frame and speedbias
  int marg_dim = 0;

  // index 大的先移�?
  for (int k = margVertexs.size() - 1; k >= 0; --k) {
    int idx = margVertexs[k]->OrderingId();
    int dim = margVertexs[k]->LocalDimension();
    //        std::cout << k << " "<<idx << std::endl;
    marg_dim += dim;
    // move the marg pose to the Hmm bottown right
    // �? row i 移动矩阵最下面
    Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);
    Eigen::MatrixXd temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
    H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;
    H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;

    // �? col i 移动矩阵最右边
    Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim);
    Eigen::MatrixXd temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
    H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols;
    H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;

    Eigen::VectorXd temp_b = b_marg.segment(idx, dim);
    Eigen::VectorXd temp_btail = b_marg.segment(idx + dim, reserve_size - idx - dim);
    b_marg.segment(idx, reserve_size - idx - dim) = temp_btail;
    b_marg.segment(reserve_size - dim, dim) = temp_b;
  }

  double eps = 1e-8;
  int m2 = marg_dim;
  int n2 = reserve_size - marg_dim;  // marg pose
  Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
  Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();

  Eigen::VectorXd bmm2 = b_marg.segment(n2, m2);
  Eigen::MatrixXd Arm = H_marg.block(0, n2, n2, m2);
  Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);
  Eigen::MatrixXd Arr = H_marg.block(0, 0, n2, n2);
  Eigen::VectorXd brr = b_marg.segment(0, n2);
  Eigen::MatrixXd tempB = Arm * Amm_inv;
  H_prior_ = Arr - tempB * Amr;
  b_prior_ = brr - tempB * bmm2;

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_prior_);
  Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
  Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

  Eigen::VectorXd S_sqrt = S.cwiseSqrt();
  Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
  Jt_prior_inv_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  err_prior_ = -Jt_prior_inv_ * b_prior_;

  MatXX J = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
  H_prior_ = J.transpose() * J;
  MatXX tmp_h = MatXX((H_prior_.array().abs() > 1e-9).select(H_prior_.array(), 0));
  H_prior_ = tmp_h;

  // std::cout << "my marg b prior: " <<b_prior_.rows()<<" norm: "<< b_prior_.norm() << std::endl;
  // std::cout << "    error prior: " <<err_prior_.norm() << std::endl;

  // remove vertex and remove edge
  for (size_t k = 0; k < margVertexs.size(); ++k) {
    RemoveVertex(margVertexs[k]);
  }

  for (auto landmarkVertex : margLandmark) {
    RemoveVertex(landmarkVertex.second);
  }

  return true;
}
}
}
