/*main*/
#include<iostream>
#include"pch.h"
#include<vector>
using namespace std;

int main()
{
	int         nn_layer = 4;               // Number of layers (4 including input and output layers)
	int         nn_input = 20;              // Number of input neurones, 5-Day time series = 20 neurons
	int         nn_hidden1 = 10;              // Number of neurones on the first hidden layer
	int         nn_hidden2 = 10;              // number on the second hidden layer
	int         nn_output = 1;
	int         hidden_layer = nn_layer - 2;
	int feature_num = 3;

	double inpt1[3] = { 1,2,3 };
	double oupt1[1] = { 1 };

	double inpt2[3] = { 3,2,1 };
	double oupt2[1] = { 0 };

	int hidden_neuron[] = {10,10};
	//string activation[] = { "ReLu", "ReLu"};
	int activation[] = { 0, 1 };
	vector<int> v = { 8,2,2,3 };
	vector<int> v1(8, 8 + 3);
	//bool s = build_net(hidden_layer, feature_num, nn_output, inpt1, oupt1, hidden_neuron);
	int a = build_net(hidden_layer, feature_num, nn_output, inpt1, oupt1, hidden_neuron, activation);
	double mse = 0;
	for (int i = 0; i < 20; i++) {
		mse = train(feature_num, nn_output, inpt1, oupt1, 2, 0.01 ,0);
		cout << mse << endl;
		mse = train(feature_num, nn_output, inpt2, oupt2, 2, 0.01, 0);
		cout << mse << endl;
	}

	//string s[] = { "er", "okt" };
	//cout << s[0] << endl;

	//double p = 0;
	//double p_hat[] = { 1 };  //output_num
	//if (s)
	//	Predict(feature_num, nn_output, inpt, p_hat);
	//release_net();
	int aa = 0;
	aa = myAdd(5, 4);
	cout << aa << endl;

	return 0;
}
