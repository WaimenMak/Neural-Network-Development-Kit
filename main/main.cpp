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
//	// Eigen 以矩阵为基本数据单元，它是一个模板类，它的前三个参数为: 数据类型，行，列
//	Matrix<float, 2, 3> matrix_23;          // 声明一个2*3的float矩阵
//	// 同时Eigen通过typedef 提供了许多内置类型，不过底层仍是Eigen::Matrix
//	// 例如Vector3d实质上是Eigen::Matrix<double,3,1>,即三维向量
//	Vector3d v_3d;
//	// 这是一样的
//	Matrix<float, 3, 1> vd_3d;
//
//	// Matrix3d 实质上是Eigen::Matrix<double,3,3>
//	Matrix3d matrix_33 = Eigen::Matrix3d::Zero();    // 初始化为零
//
//	// 如果不确定矩阵大小，可以使用动态大小的矩阵
//	Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
//	// 下面是一种更简单的写法
//	MatrixXd matrix_x;
//
//	// 下面是对Eigen矩阵的操作
//	// 输入数据(初始化）
//	matrix_23 << 1, 2, 3, 4, 5, 6;
//	// 输出
//	cout << matrix_23 << endl;
//
//	// 用()访问矩阵中的元素
//	for (int i = 0; i < 2; i++)
//		for (int j = 0; j < 3; j++)
//			cout << matrix_23(i, j) << "\t";
//
//	cout << endl;
//
//	// 矩阵和向量的相乘(实际上仍然是矩阵和矩阵)
//	v_3d << 3, 2, 1;
//	vd_3d << 4, 5, 6;
//
//	// 但是在Eigen里你不能混合两种不同类型的矩阵，像这样是错的,下面是double和float
//	// Eigen::Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;
//	// 应该显式转换
//	Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
//	cout << result << endl;
//
//	// 标准的矩阵乘法
//	Eigen::Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
//	cout << result2 << endl;
//
//	// 矩阵的维数不对，会报错
//	// Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;
//
//	// 一些矩阵运算
//	// 四则运算就不演示了，直接用+-*/即可。
//	matrix_33 = Matrix3d::Random();      // 随机数矩阵
//	cout << matrix_33 << endl;                  // 输出矩阵
//	cout << "-------------------------" << endl;
//	cout << matrix_33.transpose() << endl;      // 矩阵转置
//	cout << "-------------------------" << endl;
//	cout << matrix_33.sum() << endl;            // 各元素的和
//	cout << "-------------------------" << endl;
//	cout << matrix_33.trace() << endl;          // 矩阵的迹
//	cout << "-------------------------" << endl;
//	cout << 10 * matrix_33 << endl;             // 数乘
//	cout << "-------------------------" << endl;
//	cout << matrix_33.inverse() << endl;        // 矩阵求逆
//	cout << "-------------------------" << endl;
//	cout << matrix_33.determinant() << endl;    // 行列式
//
//	// 特征值
//	// 实对称矩阵可以保证对角化成功
//	SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
//	cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
//	cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;   // 特征值对应的特征向量列排列’
//
//	Matrix< double, MATRIX_SIZE, MATRIX_SIZE > matrix_NN;           // 声明一个MATRIX_SIZE*MATRIX_SIZE矩阵
//
//	matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);         // 矩阵初始化
//
//	Matrix< double, MATRIX_SIZE, 1> v_Nd;
//	v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);
//
//	clock_t time_stt = clock();                                            // 计时
//	// 直接求逆
//	Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
//	cout << "time use in normal inverse is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
//
//	// 通常用矩阵分解来求，例如QR分解，速度会快很多
//	time_stt = clock();
//	x = matrix_NN.colPivHouseholderQr().solve(v_Nd);                       // QR分解
//	cout << "time use in Qr decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
//	return 0;
//}