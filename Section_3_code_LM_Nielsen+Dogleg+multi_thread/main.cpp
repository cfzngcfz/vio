#include <iostream>
#include <random>
#include "problem.h"

using namespace myslam::backend;
using namespace std;

// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class CurveFittingVertex : public Vertex
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingVertex() : Vertex(3) {}                                         // abc: 三个参数， Vertex 是 3 维的 // 定义待优化的参数的维度
    virtual std::string TypeInfo() const { return "abc"; }
};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge : public Edge
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge(double x, double y) : Edge(1, 1, std::vector<std::string>{"abc"}) {
        x_ = x; //自变量
        y_ = y; //观测值
    }
    
    // 计算曲线模型误差
    virtual void ComputeResidual() override
    {
        Vec3 abc = verticies_[0]->Parameters();                                 // 估计的参数 // verticies_ 边对应的顶点（待优化的参数） defined in edge.h
        residual_(0) = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2)) - y_;  // 构建残差   // 当前观测值的残差
                                                                                // exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2))(真实值/预测值)
                                                                                // y_(观测值)
        //residual_(0) = abc(0) * x_ * x_ + abc(1) * x_ + abc(2) - y_;          // 作业1.2，修改1-3
    }

    // 计算残差对变量的雅克比
    virtual void ComputeJacobians() override
    {
        Vec3 abc = verticies_[0]->Parameters();                                 // verticies_ 边对应的顶点（待优化的参数） defined in edge.h
        Eigen::Matrix<double, 1, 3> jaco_abc;                                   // 声明 残差关于待优化参数的雅可比矩阵
                                                                                // 残差是1维的，待优化参数是3维的，所以雅克比矩阵是1*3
        
        // 雅可比矩阵的构造
        double exp_y = std::exp(abc(0) * x_ * x_ + abc(1) * x_ + abc(2));
        jaco_abc << x_ * x_ * exp_y, x_* exp_y, 1 * exp_y;
        //jaco_abc << x_ * x_, x_, 1;                                           // 作业1.2，修改2-3

        jacobians_[0] = jaco_abc;
    }
    // 返回边的类型信息
    virtual std::string TypeInfo() const override { return "CurveFittingEdge"; }
private:
    double x_, y_;  // x 值， y 值为 _measurement
};

int main()
{
    double a = 1.0, b = 2.0, c = 1.0;         // 真实参数值
    int N = 100;                          // 数据点
    double w_sigma = 1.;                 // 噪声Sigma值

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0., w_sigma);

    // 构建 problem
    Problem problem(Problem::ProblemType::GENERIC_PROBLEM);             // 通用最小二乘问题
    shared_ptr< CurveFittingVertex > vertex(new CurveFittingVertex());  // 声明 待优化的动态量/自变量

    // 设定待估计参数 a, b, c初始值
    vertex->SetParameters(Eigen::Vector3d(0., 0., 0.));                 // 自变量初始化
    // 将待估计的参数加入最小二乘问题
    problem.AddVertex(vertex);                                          // 告诉problem优化的对象是vertex

    // 构造 N 次观测
    for (int i = 0; i < N; ++i) {

        // 数据生成
        double x = i / 100.;
        double n = noise(generator);
        double y = std::exp(a * x * x + b * x + c) + n;                 // 观测值/理论值
        //double y = a*x*x + b*x + c+n;                                 // 作业1.2，修改3-3

        // 每个观测对应的残差函数
        shared_ptr< CurveFittingEdge > edge(new CurveFittingEdge(x, y)); // 声明 边(自变量，观测值)
        std::vector<std::shared_ptr<Vertex>> edge_vertex;                // 声明 边对应的顶点
        edge_vertex.push_back(vertex);                                   // 边对应的顶点，即残差有关的自变量
        edge->SetVertex(edge_vertex);                                    // 顶点对应的边 传入 边

        // 把这个残差添加到最小二乘问题
        problem.AddEdge(edge);                                           // 边 传入 问题
    }

    std::cout << "\nTest CurveFitting start..." << std::endl;
    // 使用 LM 求解
    problem.Solve(30);                                                    // 问题求解，执行30次

    std::cout << "---------" << std::endl;
    std::cout << "After optimization, we got these parameters :" << std::endl;
    std::cout << vertex->Parameters().transpose() << std::endl;
    std::cout << "ground truth: " << std::endl;
    std::cout << a << ", " << b << ", " << c << std::endl;

    return 0;
}
