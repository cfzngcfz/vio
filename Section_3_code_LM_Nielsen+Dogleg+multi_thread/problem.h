#ifndef MYSLAM_BACKEND_PROBLEM_H
#define MYSLAM_BACKEND_PROBLEM_H

#include <unordered_map>
#include <map>
#include <memory>
#include <mutex>


#include "eigen_types.h"
#include "edge.h"
#include "vertex.h"

typedef unsigned long ulong;

namespace myslam {
    namespace backend {

        class Problem {
        public:

            /**
             * ���������
             * SLAM���⻹��ͨ�õ�����
             *
             * �����SLAM������ôpose��landmark�����ֿ��ģ�Hessian��ϡ�跽ʽ�洢
             * SLAM����ֻ����һЩ�ض���Vertex��Edge
             * �����ͨ��������ôhessian�ǳ��ܵģ������û��趨ĳЩvertexΪmarginalized
             */
            enum class ProblemType {
                SLAM_PROBLEM,
                GENERIC_PROBLEM
            };
            typedef unsigned long ulong;
            //    typedef std::unordered_map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
            typedef std::map<unsigned long, std::shared_ptr<Vertex>> HashVertex;
            typedef std::unordered_map<unsigned long, std::shared_ptr<Edge>> HashEdge;
            typedef std::unordered_multimap<unsigned long, std::shared_ptr<Edge>> HashVertexIdToEdge;

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
             * ȡ�����Ż��б��ж�Ϊoutlier���ֵıߣ�����ǰ��ȥ��outlier
             * @param outlier_edges
             */
            void GetOutlierEdges(std::vector<std::shared_ptr<Edge>>& outlier_edges);

            /**
             * ��������
             * @param iterations
             * @return
             */
            void SaveCostTime(std::string filename, double SolveTime, double HessianTime);
            bool Solve(int iterations);


            /// ��Ե��һ��frame������Ϊhost��landmark
            bool Marginalize(std::shared_ptr<Vertex> frameVertex,
                const std::vector<std::shared_ptr<Vertex>>& landmarkVerticies);

            bool Marginalize(const std::shared_ptr<Vertex> frameVertex);

            //test compute prior
            void TestComputePrior();

        private:

            /// Solve��ʵ�֣���ͨ������
            bool SolveGenericProblem(int iterations);

            /// Solve��ʵ�֣���SLAM����
            bool SolveSLAMProblem(int iterations);

            /// ���ø������ordering_index
            void SetOrdering();

            /// set ordering for new vertex in slam problem
            void AddOrderingSLAM(std::shared_ptr<Vertex> v);

            /// �����H����
            void MakeHessian();
            void edge_thread(std::vector<ulong> vec_id);

            /// schur���SBA
            void SchurSBA();

            /// �����Է���
            void SolveLinearSystem();

            /// ����״̬����
            void UpdateStates();

            void RollbackStates(); // ��ʱ�� update ��в������Ҫ�˻�ȥ������

            /// ���㲢����Prior����
            void ComputePrior();

            /// �ж�һ�������Ƿ�ΪPose����
            bool IsPoseVertex(std::shared_ptr<Vertex> v);

            /// �ж�һ�������Ƿ�Ϊlandmark����
            bool IsLandmarkVertex(std::shared_ptr<Vertex> v);

            /// �������������Ҫ��������hessian�Ĵ�С
            void ResizePoseHessiansWhenAddingPose(std::shared_ptr<Vertex> v);

            /// ���ordering�Ƿ���ȷ
            bool CheckOrdering();

            void LogoutVectorSize();

            /// ��ȡĳ���������ӵ��ı�
            std::vector<std::shared_ptr<Edge>> GetConnectedEdges(std::shared_ptr<Vertex> vertex);

            /// Levenberg
            /// ����LM�㷨�ĳ�ʼLambda
            void ComputeLambdaInitLM();

            /// Hessian �Խ��߼��ϻ��߼�ȥ  Lambda
            void AddLambdatoHessianLM();

            void RemoveLambdaHessianLM();

            /// LM �㷨�������ж� Lambda ���ϴε������Ƿ���ԣ��Լ�Lambda��ô����
            bool IsGoodStepInLM();
            bool IsGoodStepInDogleg();

            /// PCG �������������
            VecX PCGSolver(const MatXX& A, const VecX& b, int maxIter);

            double currentLambda_;      // ��ǰ����������
            double currentChi_;         // ��ǰ����ʧ����ֵ�������б��ϵĲв�ƽ����
            double stopThresholdLM_;    // LM �����˳���ֵ����
            double ni_;                 // ���� Lambda ���Ŵ�С

            ProblemType problemType_;

            /// ������Ϣ����
            MatXX Hessian_;
            VecX b_;
            VecX delta_x_;

            /// ���鲿����Ϣ
            MatXX H_prior_;
            VecX b_prior_;
            MatXX Jt_prior_inv_;
            VecX err_prior_;

            /// SBA��Pose����
            MatXX H_pp_schur_;
            VecX b_pp_schur_;
            // Heesian �� Landmark �� pose ����
            MatXX H_pp_;
            VecX b_pp_;
            MatXX H_ll_;
            VecX b_ll_;

            /// all vertices
            HashVertex verticies_;

            /// all edges
            HashEdge edges_;                                  // ��¼ÿ�����ϵĲв��residual_

            // Dogleg �㷨����Ҫ�ı���
            double alpha_;
            double beta_;
            VecX h_dl_, h_sd_, h_gn_;
            double radius_;  // ������

            // multi_thread
            MatXX H_all_; // ������Դ
            VecX b_all_;  // ������Դ
            std::mutex m_H_; // ������
            std::mutex m_b_; // ������


            /// ��vertex id��ѯedge
            HashVertexIdToEdge vertexToEdge_;

            /// Ordering related
            ulong ordering_poses_ = 0;
            ulong ordering_landmarks_ = 0;
            ulong ordering_generic_ = 0;
            std::map<unsigned long, std::shared_ptr<Vertex>> idx_pose_vertices_;        // ��ordering�����pose����
            std::map<unsigned long, std::shared_ptr<Vertex>> idx_landmark_vertices_;    // ��ordering�����landmark����

            // verticies need to marg. <Ordering_id_, Vertex>
            HashVertex verticies_marg_;

            bool bDebug = false;
            double t_hessian_cost_ = 0.0;
            double t_PCGsovle_cost_ = 0.0;
        };

    }
}

#endif