#ifndef MYSLAM_BACKEND_PROBLEM_H
#define MYSLAM_BACKEND_PROBLEM_H

#include <map>
#include <memory>
#include <unordered_map>
#include <mutex>

#include "edge.h"
#include "eigen_types.h"
#include "vertex.h"

typedef unsigned long ulong;

namespace myslam {
namespace backend {

typedef unsigned long ulong;
//    typedef std::unordered_map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

class Problem {
 public:
  /**
   * 问题的类�?
   * SLAM问题还是通用的问�?
   *
   * 如果是SLAM问题那么pose和landmark是区分开的，Hessian以稀疏方式存�?
   * SLAM问题只接受一些特定的Vertex和Edge
   * 如果是通用问题那么hessian是稠密的，除非用户设定某些vertex为marginalized
   */
  enum class ProblemType { SLAM_PROBLEM, GENERIC_PROBLEM };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Problem(ProblemType problemType);

  ~Problem();

  bool AddVertex(std::shared_ptr<Vertex> vertex);

  /**
   * remove a vertex
   * @param vertex_to_remove
   */
  bool RemoveVertex(std::shared_ptr<Vertex> vertex);

  bool AddEdge(std::shared_ptr<Edge> edge);

  bool RemoveEdge(std::shared_ptr<Edge> edge);

  /**
   * 取得在优化中被判断为outlier部分的边，方便前端去除outlier
   * @param outlier_edges
   */
  void GetOutlierEdges(std::vector<std::shared_ptr<Edge>>& outlier_edges);

  /**
   * 求解此问�?
   * @param iterations
   * @return
   */
  bool Solve(int iterations = 10);

  /// 边缘化一个frame和以它为host的landmark
  bool Marginalize(std::shared_ptr<Vertex> frameVertex, const std::vector<std::shared_ptr<Vertex>>& landmarkVerticies);

  bool Marginalize(const std::shared_ptr<Vertex> frameVertex);
  bool Marginalize(const std::vector<std::shared_ptr<Vertex>> frameVertex, int pose_dim);

  MatXX GetHessianPrior() { return H_prior_; }
  VecX GetbPrior() { return b_prior_; }
  VecX GetErrPrior() { return err_prior_; }
  MatXX GetJtPrior() { return Jt_prior_inv_; }

  void SetHessianPrior(const MatXX& H) { H_prior_ = H; }
  void SetbPrior(const VecX& b) { b_prior_ = b; }
  void SetErrPrior(const VecX& b) { err_prior_ = b; }
  void SetJtPrior(const MatXX& J) { Jt_prior_inv_ = J; }

  void ExtendHessiansPriorSize(int dim);

  // test compute prior
  void TestComputePrior();

 private:
  /// Solve的实现，解通用问题
  bool SolveGenericProblem(int iterations);

  /// Solve的实现，解SLAM问题
  bool SolveSLAMProblem(int iterations);

  /// 设置各顶点的ordering_index
  void SetOrdering();

  /// set ordering for new vertex in slam problem
  void AddOrderingSLAM(std::shared_ptr<Vertex> v);

  /// 构造大H矩阵
  void MakeHessian();
  void edge_thread(std::vector<ulong> vec_id);

  /// schur求解SBA
  void SchurSBA();

  /// 解线性方�?
  void SolveLinearSystem();

  /// 更新状态变�?
  void UpdateStates();

  void RollbackStates();  // 有时�? update 后残差会变大，需要退回去，重�?

  /// 计算并更新Prior部分
  void ComputePrior();

  /// 判断一个顶点是否为Pose顶点
  bool IsPoseVertex(std::shared_ptr<Vertex> v);

  /// 判断一个顶点是否为landmark顶点
  bool IsLandmarkVertex(std::shared_ptr<Vertex> v);

  /// 在新增顶点后，需要调整几个hessian的大�?
  void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

  /// 检查ordering是否正确
  bool CheckOrdering();

  void LogoutVectorSize();

  /// 获取某个顶点连接到的�?
  std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);

  /// Levenberg
  /// 计算LM算法的初始Lambda
  void ComputeLambdaInitLM();

  /// Hessian 对角线加上或者减�?  Lambda
  void AddLambdatoHessianLM();

  void RemoveLambdaHessianLM();

  /// LM 算法中用于判�? Lambda 在上次迭代中是否可以，以及Lambda怎么缩放
  bool IsGoodStepInLM();
  bool IsGoodStepInDogleg();

  /// PCG 迭代线性求解器
  VecX PCGSolver(const MatXX& A, const VecX& b, int maxIter);

  double currentLambda_;
  double currentChi_;
  double stopThresholdLM_;  // LM 迭代退出阈值条�?
  double ni_;               //控制 Lambda 缩放大小

  ProblemType problemType_;

  /// 整个信息矩阵
  MatXX Hessian_;
  VecX b_;
  VecX delta_x_;

  /// 先验部分信息
  MatXX H_prior_;
  VecX b_prior_;
  VecX b_prior_backup_;
  VecX err_prior_backup_;

  MatXX Jt_prior_inv_;
  VecX err_prior_;

  /// SBA的Pose部分
  MatXX H_pp_schur_;
  VecX b_pp_schur_;
  // Heesian �? Landmark �? pose 部分
  MatXX H_pp_;
  VecX b_pp_;
  MatXX H_ll_;
  VecX b_ll_;

  /// all vertices
  HashVertex verticies_;

  /// all edges
  HashEdge edges_; // pair<ulong, std::shared_ptr<Edge>>

  // Dogleg 算法所需要的变量
  double alpha_;
  double beta_;
  VecX h_dl_, h_sd_, h_gn_;
  double radius_;  // 信赖�?

  // multi_thread
  MatXX H_all_; // 共享资源
  VecX b_all_;  // 共享资源
  std::mutex m_H_; // 互斥�?
  std::mutex m_b_; // 互斥�?

  double time_solve_;
  double time_makeHessian_;

  /// 由vertex id查询edge
  HashVertexIdToEdge vertexToEdge_;

  /// Ordering related
  ulong ordering_poses_ = 0;
  ulong ordering_landmarks_ = 0;
  ulong ordering_generic_ = 0;
  std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_;      // 以ordering排序的pose顶点
  std::map<unsigned long, std::shared_ptr<Vertex>> idx_landmark_vertices_;  // 以ordering排序的landmark顶点

  // verticies need to marg. <Ordering_id_, Vertex>
  HashVertex verticies_marg_;

  bool bDebug = false;
  double t_hessian_cost_ = 0.0;
  double t_PCGsovle_cost_ = 0.0;
};
}
}

#endif
