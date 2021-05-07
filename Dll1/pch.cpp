// pch.cpp: 与预编译标头对应的源文件

#include "pch.h"
#include<iostream>
#include "Nodes2.h"
#include <Eigen/Dense>              
#include <Eigen/Core>
#include<random>
#include<Eigen/Eigen>
#include<ctime>
#include"Utils.h"
#include <vector>

// 当使用预编译的头时，需要使用此源文件，编译才能成功。
using namespace std;
Nodes* net = NULL;
Nodes* output = NULL;
Input* inpt = NULL;
Const* y = NULL;
vector<Nodes*> p;
//G_D_Optimizer GD(p, 0.01);

//nn_layer: total layer - (inpt layer + oupt layer) = hidden layer
bool build_net(int & nn_layer, int & feature_num, int & nn_output, const double inputVector[], const double  outputVector[], const int neuron_num[], const int active[]) {
	int batch_num = 1;
	bool successful = false;

	MatrixXd data_x(1, feature_num), data_y(1, nn_output);
	for (int s = 0; s < batch_num; s++) {
		for (int i = 0; i < feature_num; i++) {
			data_x(0, i) = inputVector[i];
			//cout << data_x << endl;
		}

		for (int j = 0; j < nn_output; j++) {
			data_y(0, j) = outputVector[j];
			//cout << data_y << endl;
		}
	}

	inpt = new Input(data_x);
	y = new Const(data_y);


	Nodes* l1 = Fully_Conect(inpt, neuron_num[0]);
	Nodes* activate = NULL;
	if (active[0] == 0) {
		activate = new ReLu(l1);
	}
	else if (active[0] == 1) {
		activate = new Sigmoid(l1);
	}
	else if (active[0] == 2) {
		activate = new Tanh(l1);
	}
	else if (active[0] == 3) {
		activate = new Lee_Osc(l1);
	}
	else {
		activate = l1;
	}

	Nodes* temp = NULL;
	for (int layer = 1; layer < nn_layer; layer++) {
		temp = Fully_Conect(activate, neuron_num[layer]);
		if (active[layer] == 0) {
			activate = new ReLu(temp);
		}
		else if (active[layer] == 1) {
			activate = new Sigmoid(temp);
		}
		else if (active[layer] == 2) {
			activate = new Tanh(temp);
		}
		else if (active[layer] == 3) {
			activate = new Lee_Osc(temp);
		}
		else {
			activate = temp;
		}
	}


	output = Fully_Conect(activate, nn_output);


	//Nodes* l1 = Fully_Conect(inpt, 10);
	//Nodes* l2 = new ReLu(l1);
	//Nodes* l3 = Fully_Conect(l2, 10);
	//Nodes* l4 = new ReLu(l3);
	//Nodes* l5 = Fully_Conect(l4, 3);
	//Nodes* l6 = new Lee_Osc(l5);
	//Nodes* l7 = Fully_Conect(l6, nn_output);
	//output = l7;
	

	Nodes* cost = MSE(output, y);
	net = cost;
	if (cost != NULL) {
		successful = true;
		Train(net, p);
	}
	return successful;
}

void Predict(int feature_num, int nn_output, const double inputVector[], double  * y_hat) {
	int batch_num = 1;
	for (int s = 0; s < batch_num; s++) {
		for (int i = 0; i < feature_num; i++) {
			inpt->node.value(s, i) = inputVector[i];
		}

	}
	Forward(output);
	for (int j = 0; j < nn_output; j++) {
		*(y_hat+j) = output->node.value(0, j); //y_hat->prediction
	}
}

double train(int feature_num, int nn_output, const double inputVector[], const double  outputVector[], int iteration, const double & lr, const int & opt) {
	int batch_num = 1;
	for (int s = 0; s < batch_num; s++) {
		for (int i = 0; i < feature_num; i++) {
			inpt->node.value(s, i) = inputVector[i];
		}
		for (int j = 0; j < nn_output; j++) {
			//net->last_left->last_right->node.value(s, j) = outputVector[j];
			y->node.value(s, j) = outputVector[j];

		}
	}
	/*Forward(net);
	Backprop(net);*/
	if (opt == 1) {
		Adagrad_Optimizer adagrad(p, lr);
		for (int iter = 0; iter < iteration; iter++) {
			Forward(net);
			Backprop(net);
			adagrad.train();
		}
	}
	else {
		G_D_Optimizer GD(p, lr);
		for (int iter = 0; iter < iteration; iter++) {
			Forward(net);
			Backprop(net);
			GD.train();
		}
	}
	//G_D_Optimizer GD(p, lr);
	//for (int iter = 0; iter < iteration; iter++) {
	//	Forward(net);
	//	Backprop(net);
	//	GD.train();
	//}

	double cost = net->node.value(0, 0);
	return cost;
}

void release_net() {
	Finish(net);
	p.clear();
}

int myAdd(int a, int b)
{
	printf("I am here!\n");
	return a - b;
}


