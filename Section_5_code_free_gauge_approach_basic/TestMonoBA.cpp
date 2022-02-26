#include <iostream>
#include <random>
#include "vertex_inverse_depth.h"
#include "vertex_pose.h"
#include "edge_reprojection.h"
#include "problem.h"

using namespace myslam::backend;
using namespace std;

/*
 * Frame : 保存每帧的姿态和观测
 */
struct Frame {
    // 1.当前帧的位姿
    Frame(Eigen::Matrix3d R, Eigen::Vector3d t) : Rwc(R), qwc(R), twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;
    // 2.当前帧观测到的所有特征点
    unordered_map<int, Eigen::Vector3d> featurePerId; // 特征点id + 观测值/归一化像素
};

/*
 * 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
 */
void GetSimDataInWordFrame(vector<Frame>& cameraPoses, vector<Eigen::Vector3d>& points) {
    // cameraPoses: 记录第所有相机/所有帧的位姿，即平移+旋转 from camera frame to world frame
    // points:      记录所有特征点在 world frame 的空间坐标
    int featureNums = 3;  // 特征数目，假设每帧都能观测到所有的特征
    int poseNums = 3;      // 相机数目/帧数

    // 1.生成 poseNums 个 位姿
    double radius = 8;
    for (int i = 0; i < poseNums; ++i) {
        double theta = i * 2 * M_PI / (poseNums * 4); // 旋转向量
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()); // 旋转向量->旋转矩阵
        // R: 第i个相机/第i帧的旋转 from camera frame to world frame
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        // t: 第i个相机/第i帧的平移 from camera frame to world frame
        cameraPoses.push_back(Frame(R, t));
    }

    //cout << "===== 特征点 =====" << endl;
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1. / 1000.);  // 2pixel / focal
    for (int j = 0; j < featureNums; ++j) {
        // 2.生成 world frame 中的 featureNums个特征点
        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
        std::uniform_real_distribution<double> z_rand(4., 8.);
        Eigen::Vector3d Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
        // Pw: 第j个特征点的在 world frame 的空间坐标
        //cout << "第" << j << "个特征点 in world frame: " << Pw.transpose() << endl;
        points.push_back(Pw);
        
        // 3.根据每帧的位姿，将 特征点的三维空间坐标 转化为 每帧的观测值/归一化像素
        for (int i = 0; i < poseNums; ++i) {
            Eigen::Vector3d Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
            // Pc: 当前特征点 在第i个相机/第i帧 的 三维空间坐标 in camera frame
            Pc = Pc / Pc.z();  // 归一化图像平面
            Pc[0] += noise_pdf(generator);
            Pc[1] += noise_pdf(generator);
            // Pc变为: 当前特征点 在第i个相机/第i帧 的 观测值/归一化像素(第三维为1)
            //cout << "第" << i << "帧归一化像素: " << Pc.transpose() << endl;
            cameraPoses[i].featurePerId.insert(make_pair(j, Pc));
        }
    }
}

int main() {
    // 准备数据
    vector<Frame> cameras;
    vector<Eigen::Vector3d> points;
    GetSimDataInWordFrame(cameras, points); // 生成随机数据

    /*cout << "\n===== cameras =====" << endl;
    for (size_t ii = 0; ii < cameras.size(); ii++)
    {
        cout << "第" << ii << "帧的旋转\n" << cameras[ii].Rwc << endl;
        cout << "第" << ii << "帧的位移: " << cameras[ii].twc.transpose() << endl;
        for (auto kv : cameras[ii].featurePerId)
        {
            cout << "第" << ii << "帧观测到第" << kv.first << "个特征点的观测值/归一化像素: " << kv.second.transpose() << endl;
        }
    }*/


    // 外参
    Eigen::Quaterniond qic(1, 0, 0, 0); // 旋转 from camera frame to imu frame
    Eigen::Vector3d tic(0, 0, 0); // 平移 from camera frame to imu frame
    //cout << qic.matrix() << endl; // qic对应的旋转矩阵是个单位阵，且tic是个零向量，说明camera frame 和 imu frame 重叠

    // 构建 problem 实例化
    Problem problem(Problem::ProblemType::SLAM_PROBLEM); //SLAM最小二乘

    // 所有 Pose
    vector<shared_ptr<VertexPose> > vertexCams_vec;
    // shared_ptr: 允许多个指针指向同一个对象
    // vertexCams_vec: 记录所有的VertexPose，每个VertexPose，记录每个相机/帧的位姿，平移+旋转 from camera frame to world frame
    for (size_t i = 0; i < cameras.size(); ++i) {
        shared_ptr<VertexPose> vertexCam(new VertexPose()); //声明一个新的VertexPose类
        Eigen::VectorXd pose(7);
        pose << cameras[i].twc, cameras[i].qwc.x(), cameras[i].qwc.y(), cameras[i].qwc.z(), cameras[i].qwc.w(); //第i个相机的位姿，平移+旋转 from camera frame to world frame 
        vertexCam->SetParameters(pose);
        
        //// 取消注释后变成 gauge fixation approach; 注释后变成 free gauge approach
        //if (i < 2)// 如果固定前2个点
        //{
        //    vertexCam->SetFixed();
        //}

        problem.AddVertex(vertexCam); // 向problem添加位姿顶点，即部分待优化的参数/部分自变量
        vertexCams_vec.push_back(vertexCam);
    }

    cout << "\n===== VertexPose =====" << endl;
    for (size_t kk = 0; kk < vertexCams_vec.size(); kk++)
    {
        cout << "\n" << kk << " Vertex->Parameters(): " << vertexCams_vec[kk]->Parameters().transpose() << endl;     // 顶点的数值
        cout << kk << " Vertex->Dimension(): " << vertexCams_vec[kk]->Dimension() << endl;          // 顶点的输入维度
        cout << kk << " Vertex->LocalDimension(): " << vertexCams_vec[kk]->LocalDimension() << endl;// 顶点中实际参与优化的维度，即自变量
        cout << kk << " Vertex->Id(): " << vertexCams_vec[kk]->Id() << endl;                        // 顶点id
        cout << kk << " Vertex->OrderingId(): " << vertexCams_vec[kk]->OrderingId() << endl;        // Id和OrderingId的区别是什么!!!!
        cout << kk << " Vertex->TypeInfo(): " << vertexCams_vec[kk]->TypeInfo() << endl;
    }
    cout << "\n===== edge =====" << endl;

    // 所有 Point 及 edge
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0, 1.);
    double noise = 0;
    vector<double> noise_invd;                          // 以数值的形式，记录每个特征点的逆深度
    vector<shared_ptr<VertexInverseDepth> > allPoints;  // 以VertexInverseDepth的形式，记录每个特征点的逆深度 
    for (size_t j = 0; j < points.size(); ++j) {
        //假设所有特征点的起始帧为第0帧， 逆深度容易得到
        Eigen::Vector3d Pw = points[j]; // Pw: 第j个特征点的三维空间坐标 in world frame
        Eigen::Vector3d Pc = cameras[0].Rwc.transpose() * (Pw - cameras[0].twc); // Pc: 第j个特征点的三维空间坐标 in 0-th camera frame
        noise = noise_pdf(generator);
        double inverse_depth = 1. / (Pc.z() + noise); //第j个特征点的 逆深度+噪声
        // double inverse_depth = 1. / Pc.z();
        noise_invd.push_back(inverse_depth);

        // 初始化特征 vertex
        shared_ptr<VertexInverseDepth> verterxPoint(new VertexInverseDepth());
        VecX inv_d(1); // 声明一个长度为1的向量
        inv_d << inverse_depth;
        verterxPoint->SetParameters(inv_d);
        problem.AddVertex(verterxPoint); // 向problem添加(特征点的)逆深度顶点，即部分待优化的参数/部分自变量
        allPoints.push_back(verterxPoint);


        // 每个特征对应的投影误差, 第 0 帧为起始帧
        for (size_t i = 1; i < cameras.size(); ++i) { // 遍历每个相机/帧
            Eigen::Vector3d pt_0 = cameras[0].featurePerId.find(j)->second;     // 第j个特征点的观测值/归一化像素(第三维为1) in 0-th camera frame 
            Eigen::Vector3d pt_i = cameras[i].featurePerId.find(j)->second;     // 第j个特征点的观测值/归一化像素(第三维为1) in i-th camera frame 
            shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_0, pt_i));//声明一个新的EdgeReprojection类，pt_0和pt_i分别对应类定义中的pts_i_和pts_j_
            edge->SetTranslationImuFromCamera(qic, tic);                        // 向edge传入外参

            std::vector<std::shared_ptr<Vertex> > edge_vertex;                  // 记录 边关联的顶点
            edge_vertex.push_back(verterxPoint);                                // 当前特征点逆深度 对应的顶点
            edge_vertex.push_back(vertexCams_vec[0]);                           // 第0帧 对应的顶点
            edge_vertex.push_back(vertexCams_vec[i]);                           // 第i帧 对应的顶点
            edge->SetVertex(edge_vertex);                                       // 向edge传入 边关联的顶点


            cout << "\nEdge -> Id() = " << edge->Id() << endl;                  // 边id
            cout << "Edge->NumVertices() = " << edge->NumVertices() << endl;    // 边关联的顶点数量
            for (auto vertex : edge->Verticies())                               // 边关联的所有顶点
            {
                cout << vertex->Id() << ", ";
            }
            cout << endl;
            //for (auto jacob : edge->Jacobians())                                // 边关于连通顶点的雅可比
            //{
            //    cout << jacob << endl;                                          // 由于暂未执行计算雅可比，所以没有输出
            //}
            cout << "Edge->Residual():\n" << edge->Residual() << endl;          // 残差
            cout << "Edge->Information():\n" << edge->Information() << endl;    // 信息矩阵
            cout << "Edge->Observation(): " << edge->Observation() << endl;     // 观测信息
            cout << "Edge->OrderingId(): " << edge->OrderingId() << endl;
            cout << "Edge->TypeInfo(): " << edge->TypeInfo() << endl;

            problem.AddEdge(edge);
        }
    }

    cout << "\n===== VertexInverseDepth =====" << endl;
    for (size_t kk = 0; kk < allPoints.size(); kk++)
    {
        cout << "\n" << kk << " Vertex->Parameters(): " << allPoints[kk]->Parameters().transpose() << ", noise_invd = " << noise_invd[kk] << endl;     // 顶点的数值
        cout << kk << " Vertex->Dimension(): " << allPoints[kk]->Dimension() << endl;          // 顶点的输入维度
        cout << kk << " Vertex->LocalDimension(): " << allPoints[kk]->LocalDimension() << endl;// 顶点中实际参与优化的维度，即自变量
        cout << kk << " Vertex->Id(): " << allPoints[kk]->Id() << endl;                        // 顶点id
        cout << kk << " Vertex->OrderingId(): " << allPoints[kk]->OrderingId() << endl;
        cout << kk << " Vertex->TypeInfo(): " << allPoints[kk]->TypeInfo() << endl;
    }



    problem.Solve(2); //一共迭代5次

    std::cout << "\nCompare MonoBA results after opt..." << std::endl;
    for (size_t k = 0; k < allPoints.size(); k += 1) {
        /*std::cout << "特征点 " << k << " 逆深度: 理论值 = " << 1. / points[k].z() << ", 初始值 = "
            << noise_invd[k] << ", 最优后 = " << allPoints[k]->Parameters() << std::endl;*/
        std::cout << "after opt, point " << k << " : gt " << 1. / points[k].z() << " ,noise "
            << noise_invd[k] << " ,opt " << allPoints[k]->Parameters() << std::endl;
    }
    std::cout << "------------ pose translation ----------------" << std::endl;
    for (int i = 0; i < vertexCams_vec.size(); ++i) {
        std::cout << "translation after opt: " << i << " :" << vertexCams_vec[i]->Parameters().head(3).transpose() << " || gt: " << cameras[i].twc.transpose() << std::endl;
    }
    /// 优化完成后，第一帧相机的 pose 平移（x,y,z）不再是原点 0,0,0. 说明向零空间发生了漂移。
    /// 解决办法： fix 第一帧和第二帧，固定 7 自由度。 或者加上非常大的先验值。

    // problem.TestMarginalize(); // 练习: 矩阵内元素移动、merg & 舒尔补



    return 0;
}

