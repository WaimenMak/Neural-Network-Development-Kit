#pragma once
#include <cstddef>
#include <Eigen/Dense>              
#include <Eigen/Core>
#include <math.h>
//#include "Nodes2.h"
#include "Nodes.h"
#include <queue>
#include <vector>

using namespace Eigen;
using namespace std;

//delete
void Finish(Nodes * root) {
	if (root == NULL)
		return;

	Finish(root->last_left);
	Finish(root->last_right);
	delete(root);
	return;
}

MatrixXd chain_rule(MatrixXd& par_1, MatrixXd& par_2, char node, const bool & dot) {  //reference，没有重新copy参数

	if (par_1.cols() == par_2.cols() && par_1.rows() == par_2.rows() && !dot)   //little bugs
		return par_1.cwiseProduct(par_2);

	else if (node == 'l')
		return par_1 * par_2;

	else if (node == 'r')
		return par_2 * par_1;

	return par_1;
}

Nodes* Fully_Conect(Nodes* left_node, int output_num) {

	// Gaussian distribution
	static default_random_engine e(time(0));
	static normal_distribution<double> n(0, 0.01);
	MatrixXd w = MatrixXd::Zero(left_node->node.value.cols(), output_num).unaryExpr([](double dummy) {return n(e); });

	Variable * weight = new Variable(w);

	Dot * w_x = new Dot(left_node, weight);

	w_x->output_val();

	MatrixXd b = MatrixXd::Zero(1, w_x->node.value.cols()).unaryExpr([](double dummy) {return n(e); });

	Variable * bias = new Variable(b);

	Add * output = new Add(w_x, bias);
	output->output_val();

	return output;
}


Nodes * Forward(Nodes * root) { //postOrder
	if (root == NULL) {
		return root;
	}
	Forward(root->last_left);
	Forward(root->last_right);
	root->output_val();

	return root;
}



Nodes * MSE(Nodes * output, Nodes * y) {
	Nodes * loss;   //local parameter
	Minus * m = new Minus(output, y);
	F_Norm * F = new F_Norm(m);
	loss = F;

	return loss;
}


void Train(Nodes * root, vector<Nodes *> & param) {
	if (root == NULL)
		return;

	if (root->node.BN)
		root->node.test = false;

	if (root->node.need_update)
		param.push_back(root);

	Train(root->last_left, param);
	Train(root->last_right, param);

	return;
}



void Backprop(Nodes* root) {
	queue<Nodes*> Q;
	int q_size = 0;
	Nodes* temp = NULL;
	if (root == NULL || (root->last_left == NULL && root->last_right == NULL))
		return;

	Q.push(root);

	while (!Q.empty()) {
		q_size = Q.size();
		for (int i = 0; i < q_size; i++) {
			temp = Q.front();
			Q.pop();
			if (temp->node.grad.rows() == 0) {

				temp->node.grad = MatrixXd::Ones(temp->last_left->node.value.rows(), temp->last_left->node.value.cols());   // different from python ver
				temp->compute_gradient();

				if (temp->last_left != NULL && temp->last_left->node.require_grad) {
					temp->last_left->node.grad = temp->last_left->node.sub_grad;
					Q.push(temp->last_left);
				}
				if (temp->last_right != NULL && temp->last_right->node.require_grad) {
					temp->last_right->node.grad = temp->last_right->node.sub_grad;
					Q.push(temp->last_right);
				}
			}
			else {
				temp->compute_gradient();

				if (temp->last_left != NULL && temp->last_left->node.require_grad) {
					temp->last_left->node.grad = chain_rule(temp->node.grad, temp->last_left->node.sub_grad, 'l', temp->node.dot);
					Q.push(temp->last_left);
				}
				if (temp->last_right != NULL && temp->last_right->node.require_grad) {
					temp->last_right->node.grad = chain_rule(temp->node.grad, temp->last_right->node.sub_grad, 'r', temp->node.dot);
					Q.push(temp->last_right);

				}
			}

		}

	}
}

class Optimizer {
public:
	double lr;  //learning rate
	vector<Nodes*> p;

	virtual void train() {
		return;
	}
};

class G_D_Optimizer : public Optimizer {
public:
	G_D_Optimizer(const vector<Nodes*> & param, double learning_rate) {
		lr = learning_rate;
		p = param;
	}
	~G_D_Optimizer() {

	}

	void train() {
		for (unsigned int i = 0; i < p.size(); i++) {
			if (p[i]->node.value.rows() == p[i]->node.grad.rows() && p[i]->node.value.cols() == p[i]->node.grad.cols())
				p[i]->node.value = p[i]->node.value - lr * p[i]->node.grad;
			else
				p[i]->node.value = p[i]->node.value - lr * p[i]->node.grad.colwise().sum();
		}
		return;
	}
};

class Adagrad_Optimizer : public Optimizer {
public:
	vector<MatrixXd> second_derivative;

	Adagrad_Optimizer(const vector<Nodes*> & param, double learning_rate) {
		lr = learning_rate;
		p = param;
		MatrixXd temp;
		for (unsigned int i = 0; i < param.size(); i++) {
			temp = MatrixXd::Zero(param[i]->node.value.rows(), param[i]->node.value.cols());
			second_derivative.push_back(temp);
		}
	}
	~Adagrad_Optimizer() {

	}

	void train() {
		MatrixXd g_w, g_b, d_b;    //g_w, g_b are the denominator, d_b the derivative of b, sum over b.grad
		for (unsigned int i = 0; i < p.size(); i++) {
			if (p[i]->node.value.rows() == p[i]->node.grad.rows() && p[i]->node.value.cols() == p[i]->node.grad.cols()) {
				second_derivative[i].array() += p[i]->node.grad.array().cwiseAbs2();
				g_w = (second_derivative[i].array() + 1e-10).sqrt();
				p[i]->node.value = p[i]->node.value - lr * p[i]->node.grad.cwiseQuotient(g_w);
			}
			else {
				d_b = p[i]->node.grad.colwise().sum();  //d_L/d_b
				second_derivative[i].array() += d_b.array().cwiseAbs2();
				g_b = (second_derivative[i].array() + 1e-10).sqrt();
				p[i]->node.value = p[i]->node.value - lr * d_b.cwiseQuotient(g_b);
			}
				
		}
		return;
	}
};
