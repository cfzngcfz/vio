#include <iostream>
#include <random>
#include "problem.h"

using namespace myslam::backend;
using namespace std;

// ����ģ�͵Ķ��㣬ģ��������Ż�����ά�Ⱥ���������
class CurveFittingVertex : public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingVertex() : Vertex(3) {}                                         // abc: ���������� Vertex �� 3 ά�� // ������Ż��Ĳ�����ά��
    virtual std::string TypeInfo() const { return "abc"; }
};

// ���ģ�� ģ��������۲�ֵά�ȣ����ͣ����Ӷ�������
class CurveFittingEdge : public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge(double x, double y) : Edge(1, 1, std::vector<std::string>{"abc"}) {
        x_ = x; //�Ա���
        y_ = y; //�۲�ֵ
    }
    
    // ��������ģ�����
    virtual void ComputeResidual() override
    {
        Vec3 abc = verticies_[0]->Parameters();                                 // ���ƵĲ��� // verticies_ �߶�Ӧ�Ķ��㣨���Ż��Ĳ����� defined in edge.h
        residual_(0) = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2)) - y_;  // �����в�   // ��ǰ�۲�ֵ�Ĳв�
                                                                                // exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2))(��ʵֵ/Ԥ��ֵ)
                                                                                // y_(�۲�ֵ)
        //residual_(0) = abc(0) * x_ * x_ + abc(1) * x_ + abc(2) - y_;          // ��ҵ1.2���޸�1-3
    }

    // ����в�Ա������ſ˱�
    virtual void ComputeJacobians() override
    {
        Vec3 abc = verticies_[0]->Parameters();                                 // verticies_ �߶�Ӧ�Ķ��㣨���Ż��Ĳ����� defined in edge.h
        Eigen::Matrix<double, 1, 3> jaco_abc;                                   // ���� �в���ڴ��Ż��������ſɱȾ���
                                                                                // �в���1ά�ģ����Ż�������3ά�ģ������ſ˱Ⱦ�����1*3
        
        // �ſɱȾ���Ĺ���
        double exp_y = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2));
        jaco_abc << x_ * x_ * exp_y, x_* exp_y, 1 * exp_y;
        //jaco_abc << x_ * x_, x_, 1;                                           // ��ҵ1.2���޸�2-3

        jacobians_[0] = jaco_abc;
    }
    // ���رߵ�������Ϣ
    virtual std::string TypeInfo() const override { return "CurveFittingEdge"; }
private:
    double x_, y_;  // x ֵ�� y ֵΪ _measurement
};

int main()
{
    double a = 1.0, b = 2.0, c = 1.0;         // ��ʵ����ֵ
    int N = 100;                          // ���ݵ�
    double w_sigma = 1.;                 // ����Sigmaֵ

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0., w_sigma);

    // ���� problem
    Problem problem(Problem::ProblemType::GENERIC_PROBLEM);             // ͨ����С��������
    shared_ptr< CurveFittingVertex > vertex(new CurveFittingVertex());  // ���� ���Ż��Ķ�̬��/�Ա���

    // �趨�����Ʋ��� a, b, c��ʼֵ
    vertex->SetParameters(Eigen::Vector3d(0., 0., 0.));                 // �Ա�����ʼ��
    // �������ƵĲ���������С��������
    problem.AddVertex(vertex);                                          // ����problem�Ż��Ķ�����vertex

    // ���� N �ι۲�
    for (int i = 0; i < N; ++i) {

        // ��������
        double x = i / 100.;
        double n = noise(generator);
        double y = std::exp(a * x * x + b * x + c) + n;                 // �۲�ֵ/����ֵ
        //double y = a*x*x + b*x + c+n;                                 // ��ҵ1.2���޸�3-3

        // ÿ���۲��Ӧ�Ĳв��
        shared_ptr< CurveFittingEdge > edge(new CurveFittingEdge(x, y)); // ���� ��(�Ա������۲�ֵ)
        std::vector<std::shared_ptr<Vertex>> edge_vertex;                // ���� �߶�Ӧ�Ķ���
        edge_vertex.push_back(vertex);                                   // �߶�Ӧ�Ķ��㣬���в��йص��Ա���
        edge->SetVertex(edge_vertex);                                    // �����Ӧ�ı� ���� ��

        // ������в���ӵ���С��������
        problem.AddEdge(edge);                                           // �� ���� ����
    }

    std::cout << "\nTest CurveFitting start..." << std::endl;
    // ʹ�� LM ���
    problem.Solve(30);                                                    // ������⣬ִ��30��

    std::cout << "---------" << std::endl;
    std::cout << "After optimization, we got these parameters :" << std::endl;
    std::cout << vertex->Parameters().transpose() << std::endl;
    std::cout << "ground truth: " << std::endl;
    std::cout << a << ", " << b << ", " << c << std::endl;

    return 0;
}
