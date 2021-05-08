#include<iostream>
//#include "Nodes2.h"
#include "Nodes.h"
#include <Eigen/Dense>              
#include <Eigen/Core>
#include<random>
#include<Eigen/Eigen>
#include<ctime>
#include"Utils.h"
#include <vector>

using namespace std;
using namespace Eigen;


int main()
{
	MatrixXd data_x(17,3), data_y(17,1);
	data_y << 1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1;
	data_x << 1, 2, 3,
		      8, 12, 20,
		      4, 6, 8,
		      0, 5, 11,
		      1, 2, 1,
		      8, 10, 7,
		      7, 10, 7,
		      2, 3, 1,
		      8, 7, 6,
		      20, 10,1,
		      3, 2, 1,
		      9, 4, 3,
		      7, 6, 5,
		      5, 4, 5,
		      2, 1, 6,
		      20, 12, 1,
		      8, 2, 10;
	Input * inpt = new Input(data_x);
	Const * y = new Const(data_y);

	//debug
	//MatrixXd s(17, 1);
	//s = data_y;
	//s = 1 - ((data_y.array().exp() - (-1 * data_y.array()).exp()) / (data_y.array().exp() + (-1 * data_y.array()).exp())).cwiseAbs2();
	//cout << s << endl;

	Nodes* l1 = Fully_Conect(inpt, 4);
	Nodes* l2 = new ReLu(l1);
	Nodes* l3 = Fully_Conect(l2, 3);
	//Nodes* l4 = new ReLu(l3);

	//Nodes* l5 = Fully_Conect(l4, 3);
	//Nodes* l6 = new Lee_Osc(l5);
	//Nodes* l6 = new Sigmoid(l5);
	//Nodes* l6 = new Tanh(l5);
	Nodes* l6 = new Sigmoid(l3);
	Nodes* l7 = Fully_Conect(l6, 1);



	Nodes* cost = MSE(l7, y);
	Forward(cost);
	vector<Nodes*> p;
	Train(cost, p);
	G_D_Optimizer GD(p, 0.01);
	//Adagrad_Optimizer adagrad(p, 0.01);
	//cout << cost->node.value << endl;
	int epoch = 200;

	//Train
	for (int ep = 0; ep < epoch; ep++) {

		for (int j = 0; j < 20; j++) {
			Forward(cost);
			Backprop(cost);
			GD.train();
			//adagrad.train();
		}
		if (ep % 10 == 0)
			cout << cost->node.value << endl;
	}

	//cout<<L->node.value<<endl;
	//Test
	MatrixXd x_test(2, 3);
	x_test << 1, 2, 3,
		      8, 10,7;
	inpt->node.value = x_test;

	Forward(l7);
	printf("Test:\n");
	cout << l7->node.value << endl;

	Finish(cost);
	printf("Finish!");
	//MatrixXd m(2, 2), result(2,2);
	//m << 0, 2, 3, -4;
	////m += m;
	//m.array() += m.array();
	//result = m;
	//cout << result;
}






//int main()
//{
//	// Eigen �Ծ���Ϊ�������ݵ�Ԫ������һ��ģ���࣬����ǰ��������Ϊ: �������ͣ��У���
//	Matrix<float, 2, 3> matrix_23;          // ����һ��2*3��float����
//	// ͬʱEigenͨ��typedef �ṩ������������ͣ������ײ�����Eigen::Matrix
//	// ����Vector3dʵ������Eigen::Matrix<double,3,1>,����ά����
//	Vector3d v_3d;
//	// ����һ����
//	Matrix<float, 3, 1> vd_3d;
//
//	// Matrix3d ʵ������Eigen::Matrix<double,3,3>
//	Matrix3d matrix_33 = Eigen::Matrix3d::Zero();    // ��ʼ��Ϊ��
//
//	// �����ȷ�������С������ʹ�ö�̬��С�ľ���
//	Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
//	// ������һ�ָ��򵥵�д��
//	MatrixXd matrix_x;
//
//	// �����Ƕ�Eigen����Ĳ���
//	// ��������(��ʼ����
//	matrix_23 << 1, 2, 3, 4, 5, 6;
//	// ���
//	cout << matrix_23 << endl;
//
//	// ��()���ʾ����е�Ԫ��
//	for (int i = 0; i < 2; i++)
//		for (int j = 0; j < 3; j++)
//			cout << matrix_23(i, j) << "\t";
//
//	cout << endl;
//
//	// ��������������(ʵ������Ȼ�Ǿ���;���)
//	v_3d << 3, 2, 1;
//	vd_3d << 4, 5, 6;
//
//	// ������Eigen���㲻�ܻ�����ֲ�ͬ���͵ľ����������Ǵ��,������double��float
//	// Eigen::Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;
//	// Ӧ����ʽת��
//	Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
//	cout << result << endl;
//
//	// ��׼�ľ���˷�
//	Eigen::Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
//	cout << result2 << endl;
//
//	// �����ά�����ԣ��ᱨ��
//	// Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;
//
//	// һЩ��������
//	// ��������Ͳ���ʾ�ˣ�ֱ����+-*/���ɡ�
//	matrix_33 = Matrix3d::Random();      // ���������
//	cout << matrix_33 << endl;                  // �������
//	cout << "-------------------------" << endl;
//	cout << matrix_33.transpose() << endl;      // ����ת��
//	cout << "-------------------------" << endl;
//	cout << matrix_33.sum() << endl;            // ��Ԫ�صĺ�
//	cout << "-------------------------" << endl;
//	cout << matrix_33.trace() << endl;          // ����ļ�
//	cout << "-------------------------" << endl;
//	cout << 10 * matrix_33 << endl;             // ����
//	cout << "-------------------------" << endl;
//	cout << matrix_33.inverse() << endl;        // ��������
//	cout << "-------------------------" << endl;
//	cout << matrix_33.determinant() << endl;    // ����ʽ
//
//	// ����ֵ
//	// ʵ�Գƾ�����Ա�֤�Խǻ��ɹ�
//	SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
//	cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
//	cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;   // ����ֵ��Ӧ���������������С�
//
//	Matrix< double, MATRIX_SIZE, MATRIX_SIZE > matrix_NN;           // ����һ��MATRIX_SIZE*MATRIX_SIZE����
//
//	matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);         // �����ʼ��
//
//	Matrix< double, MATRIX_SIZE, 1> v_Nd;
//	v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);
//
//	clock_t time_stt = clock();                                            // ��ʱ
//	// ֱ������
//	Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
//	cout << "time use in normal inverse is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
//
//	// ͨ���þ���ֽ���������QR�ֽ⣬�ٶȻ��ܶ�
//	time_stt = clock();
//	x = matrix_NN.colPivHouseholderQr().solve(v_Nd);                       // QR�ֽ�
//	cout << "time use in Qr decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
//	return 0;
//}