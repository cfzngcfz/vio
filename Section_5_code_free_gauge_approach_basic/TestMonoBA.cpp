#include <iostream>
#include <random>
#include "vertex_inverse_depth.h"
#include "vertex_pose.h"
#include "edge_reprojection.h"
#include "problem.h"

using namespace myslam::backend;
using namespace std;

/*
 * Frame : ����ÿ֡����̬�͹۲�
 */
struct Frame {
    // 1.��ǰ֡��λ��
    Frame(Eigen::Matrix3d R, Eigen::Vector3d t) : Rwc(R), qwc(R), twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;
    // 2.��ǰ֡�۲⵽������������
    unordered_map<int, Eigen::Vector3d> featurePerId; // ������id + �۲�ֵ/��һ������
};

/*
 * ������������ϵ�µ���������: �����̬, ������, �Լ�ÿ֡�۲�
 */
void GetSimDataInWordFrame(vector<Frame>& cameraPoses, vector<Eigen::Vector3d>& points) {
    // cameraPoses: ��¼���������/����֡��λ�ˣ���ƽ��+��ת from camera frame to world frame
    // points:      ��¼������������ world frame �Ŀռ�����
    int featureNums = 3;  // ������Ŀ������ÿ֡���ܹ۲⵽���е�����
    int poseNums = 3;      // �����Ŀ/֡��

    // 1.���� poseNums �� λ��
    double radius = 8;
    for (int i = 0; i < poseNums; ++i) {
        double theta = i * 2 * M_PI / (poseNums * 4); // ��ת����
        // �� z�� ��ת
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()); // ��ת����->��ת����
        // R: ��i�����/��i֡����ת from camera frame to world frame
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        // t: ��i�����/��i֡��ƽ�� from camera frame to world frame
        cameraPoses.push_back(Frame(R, t));
    }

    //cout << "===== ������ =====" << endl;
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1. / 1000.);  // 2pixel / focal
    for (int j = 0; j < featureNums; ++j) {
        // 2.���� world frame �е� featureNums��������
        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
        std::uniform_real_distribution<double> z_rand(4., 8.);
        Eigen::Vector3d Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
        // Pw: ��j����������� world frame �Ŀռ�����
        //cout << "��" << j << "�������� in world frame: " << Pw.transpose() << endl;
        points.push_back(Pw);
        
        // 3.����ÿ֡��λ�ˣ��� ���������ά�ռ����� ת��Ϊ ÿ֡�Ĺ۲�ֵ/��һ������
        for (int i = 0; i < poseNums; ++i) {
            Eigen::Vector3d Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);
            // Pc: ��ǰ������ �ڵ�i�����/��i֡ �� ��ά�ռ����� in camera frame
            Pc = Pc / Pc.z();  // ��һ��ͼ��ƽ��
            Pc[0] += noise_pdf(generator);
            Pc[1] += noise_pdf(generator);
            // Pc��Ϊ: ��ǰ������ �ڵ�i�����/��i֡ �� �۲�ֵ/��һ������(����άΪ1)
            //cout << "��" << i << "֡��һ������: " << Pc.transpose() << endl;
            cameraPoses[i].featurePerId.insert(make_pair(j, Pc));
        }
    }
}

int main() {
    // ׼������
    vector<Frame> cameras;
    vector<Eigen::Vector3d> points;
    GetSimDataInWordFrame(cameras, points); // �����������

    /*cout << "\n===== cameras =====" << endl;
    for (size_t ii = 0; ii < cameras.size(); ii++)
    {
        cout << "��" << ii << "֡����ת\n" << cameras[ii].Rwc << endl;
        cout << "��" << ii << "֡��λ��: " << cameras[ii].twc.transpose() << endl;
        for (auto kv : cameras[ii].featurePerId)
        {
            cout << "��" << ii << "֡�۲⵽��" << kv.first << "��������Ĺ۲�ֵ/��һ������: " << kv.second.transpose() << endl;
        }
    }*/


    // ���
    Eigen::Quaterniond qic(1, 0, 0, 0); // ��ת from camera frame to imu frame
    Eigen::Vector3d tic(0, 0, 0); // ƽ�� from camera frame to imu frame
    //cout << qic.matrix() << endl; // qic��Ӧ����ת�����Ǹ���λ����tic�Ǹ���������˵��camera frame �� imu frame �ص�

    // ���� problem ʵ����
    Problem problem(Problem::ProblemType::SLAM_PROBLEM); //SLAM��С����

    // ���� Pose
    vector<shared_ptr<VertexPose> > vertexCams_vec;
    // shared_ptr: ������ָ��ָ��ͬһ������
    // vertexCams_vec: ��¼���е�VertexPose��ÿ��VertexPose����¼ÿ�����/֡��λ�ˣ�ƽ��+��ת from camera frame to world frame
    for (size_t i = 0; i < cameras.size(); ++i) {
        shared_ptr<VertexPose> vertexCam(new VertexPose()); //����һ���µ�VertexPose��
        Eigen::VectorXd pose(7);
        pose << cameras[i].twc, cameras[i].qwc.x(), cameras[i].qwc.y(), cameras[i].qwc.z(), cameras[i].qwc.w(); //��i�������λ�ˣ�ƽ��+��ת from camera frame to world frame 
        vertexCam->SetParameters(pose);
        
        //// ȡ��ע�ͺ��� gauge fixation approach; ע�ͺ��� free gauge approach
        //if (i < 2)// ����̶�ǰ2����
        //{
        //    vertexCam->SetFixed();
        //}

        problem.AddVertex(vertexCam); // ��problem���λ�˶��㣬�����ִ��Ż��Ĳ���/�����Ա���
        vertexCams_vec.push_back(vertexCam);
    }

    cout << "\n===== VertexPose =====" << endl;
    for (size_t kk = 0; kk < vertexCams_vec.size(); kk++)
    {
        cout << "\n" << kk << " Vertex->Parameters(): " << vertexCams_vec[kk]->Parameters().transpose() << endl;     // �������ֵ
        cout << kk << " Vertex->Dimension(): " << vertexCams_vec[kk]->Dimension() << endl;          // ���������ά��
        cout << kk << " Vertex->LocalDimension(): " << vertexCams_vec[kk]->LocalDimension() << endl;// ������ʵ�ʲ����Ż���ά�ȣ����Ա���
        cout << kk << " Vertex->Id(): " << vertexCams_vec[kk]->Id() << endl;                        // ����id
        cout << kk << " Vertex->OrderingId(): " << vertexCams_vec[kk]->OrderingId() << endl;        // Id��OrderingId��������ʲô!!!!
        cout << kk << " Vertex->TypeInfo(): " << vertexCams_vec[kk]->TypeInfo() << endl;
    }
    cout << "\n===== edge =====" << endl;

    // ���� Point �� edge
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0, 1.);
    double noise = 0;
    vector<double> noise_invd;                          // ����ֵ����ʽ����¼ÿ��������������
    vector<shared_ptr<VertexInverseDepth> > allPoints;  // ��VertexInverseDepth����ʽ����¼ÿ�������������� 
    for (size_t j = 0; j < points.size(); ++j) {
        //�����������������ʼ֡Ϊ��0֡�� ��������׵õ�
        Eigen::Vector3d Pw = points[j]; // Pw: ��j�����������ά�ռ����� in world frame
        Eigen::Vector3d Pc = cameras[0].Rwc.transpose() * (Pw - cameras[0].twc); // Pc: ��j�����������ά�ռ����� in 0-th camera frame
        noise = noise_pdf(generator);
        double inverse_depth = 1. / (Pc.z() + noise); //��j��������� �����+����
        // double inverse_depth = 1. / Pc.z();
        noise_invd.push_back(inverse_depth);

        // ��ʼ������ vertex
        shared_ptr<VertexInverseDepth> verterxPoint(new VertexInverseDepth());
        VecX inv_d(1); // ����һ������Ϊ1������
        inv_d << inverse_depth;
        verterxPoint->SetParameters(inv_d);
        problem.AddVertex(verterxPoint); // ��problem���(�������)����ȶ��㣬�����ִ��Ż��Ĳ���/�����Ա���
        allPoints.push_back(verterxPoint);


        // ÿ��������Ӧ��ͶӰ���, �� 0 ֡Ϊ��ʼ֡
        for (size_t i = 1; i < cameras.size(); ++i) { // ����ÿ�����/֡
            Eigen::Vector3d pt_0 = cameras[0].featurePerId.find(j)->second;     // ��j��������Ĺ۲�ֵ/��һ������(����άΪ1) in 0-th camera frame 
            Eigen::Vector3d pt_i = cameras[i].featurePerId.find(j)->second;     // ��j��������Ĺ۲�ֵ/��һ������(����άΪ1) in i-th camera frame 
            shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_0, pt_i));//����һ���µ�EdgeReprojection�࣬pt_0��pt_i�ֱ��Ӧ�ඨ���е�pts_i_��pts_j_
            edge->SetTranslationImuFromCamera(qic, tic);                        // ��edge�������

            std::vector<std::shared_ptr<Vertex> > edge_vertex;                  // ��¼ �߹����Ķ���
            edge_vertex.push_back(verterxPoint);                                // ��ǰ����������� ��Ӧ�Ķ���
            edge_vertex.push_back(vertexCams_vec[0]);                           // ��0֡ ��Ӧ�Ķ���
            edge_vertex.push_back(vertexCams_vec[i]);                           // ��i֡ ��Ӧ�Ķ���
            edge->SetVertex(edge_vertex);                                       // ��edge���� �߹����Ķ���


            cout << "\nEdge -> Id() = " << edge->Id() << endl;                  // ��id
            cout << "Edge->NumVertices() = " << edge->NumVertices() << endl;    // �߹����Ķ�������
            for (auto vertex : edge->Verticies())                               // �߹��������ж���
            {
                cout << vertex->Id() << ", ";
            }
            cout << endl;
            //for (auto jacob : edge->Jacobians())                                // �߹�����ͨ������ſɱ�
            //{
            //    cout << jacob << endl;                                          // ������δִ�м����ſɱȣ�����û�����
            //}
            cout << "Edge->Residual():\n" << edge->Residual() << endl;          // �в�
            cout << "Edge->Information():\n" << edge->Information() << endl;    // ��Ϣ����
            cout << "Edge->Observation(): " << edge->Observation() << endl;     // �۲���Ϣ
            cout << "Edge->OrderingId(): " << edge->OrderingId() << endl;
            cout << "Edge->TypeInfo(): " << edge->TypeInfo() << endl;

            problem.AddEdge(edge);
        }
    }

    cout << "\n===== VertexInverseDepth =====" << endl;
    for (size_t kk = 0; kk < allPoints.size(); kk++)
    {
        cout << "\n" << kk << " Vertex->Parameters(): " << allPoints[kk]->Parameters().transpose() << ", noise_invd = " << noise_invd[kk] << endl;     // �������ֵ
        cout << kk << " Vertex->Dimension(): " << allPoints[kk]->Dimension() << endl;          // ���������ά��
        cout << kk << " Vertex->LocalDimension(): " << allPoints[kk]->LocalDimension() << endl;// ������ʵ�ʲ����Ż���ά�ȣ����Ա���
        cout << kk << " Vertex->Id(): " << allPoints[kk]->Id() << endl;                        // ����id
        cout << kk << " Vertex->OrderingId(): " << allPoints[kk]->OrderingId() << endl;
        cout << kk << " Vertex->TypeInfo(): " << allPoints[kk]->TypeInfo() << endl;
    }



    problem.Solve(2); //һ������5��

    std::cout << "\nCompare MonoBA results after opt..." << std::endl;
    for (size_t k = 0; k < allPoints.size(); k += 1) {
        /*std::cout << "������ " << k << " �����: ����ֵ = " << 1. / points[k].z() << ", ��ʼֵ = "
            << noise_invd[k] << ", ���ź� = " << allPoints[k]->Parameters() << std::endl;*/
        std::cout << "after opt, point " << k << " : gt " << 1. / points[k].z() << " ,noise "
            << noise_invd[k] << " ,opt " << allPoints[k]->Parameters() << std::endl;
    }
    std::cout << "------------ pose translation ----------------" << std::endl;
    for (int i = 0; i < vertexCams_vec.size(); ++i) {
        std::cout << "translation after opt: " << i << " :" << vertexCams_vec[i]->Parameters().head(3).transpose() << " || gt: " << cameras[i].twc.transpose() << std::endl;
    }
    /// �Ż���ɺ󣬵�һ֡����� pose ƽ�ƣ�x,y,z��������ԭ�� 0,0,0. ˵������ռ䷢����Ư�ơ�
    /// ����취�� fix ��һ֡�͵ڶ�֡���̶� 7 ���ɶȡ� ���߼��Ϸǳ��������ֵ��

    // problem.TestMarginalize(); // ��ϰ: ������Ԫ���ƶ���merg & �����



    return 0;
}

