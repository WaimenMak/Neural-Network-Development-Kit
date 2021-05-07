#pragma once
#include <cstddef>        // for NULL
#include <Eigen/Dense>              
#include <Eigen/Core>
#include <math.h>


struct Node {

    Eigen::MatrixXd  value;      //MatrixXd is class and value is an object, different from the python version,(pointer)
    Eigen::MatrixXd grad;
    Eigen::MatrixXd sub_grad;
    bool require_grad;
    bool need_update;
    bool BN;
    bool dot;
    bool test;

};

class Nodes {
public:
    Nodes* last_right;
    Nodes* last_left;
    Nodes* next;
    Node  node;

    Nodes(void) {
        //root = &node;
        last_right = NULL;
        last_left = NULL;
        next = NULL;
        node.require_grad = true;
        node.need_update = false;
        node.BN = false;
        node.dot = false;
        node.test = false;

    }
    ~Nodes(void) {}

    virtual Eigen::MatrixXd output_val() {
        return node.value;
    }

    virtual Eigen::MatrixXd compute_gradient() {
        return node.grad;
    }

};

class Input : public Nodes {
public:
    //Node * root;
    //template <typename mat>
    Input(Eigen::MatrixXd  input) {

        node.value = input;
    }

    ~Input(void) {}

    Eigen::MatrixXd output_val()
    {
        return node.value;
    }

    Eigen::MatrixXd compute_gradient() {
        return node.grad;
    }

};

class Const : public Nodes {
public:
    Const(const Eigen::MatrixXd X) {
        node.value = X;
        node.require_grad = false;
    }
    ~Const() {

    }
    Eigen::MatrixXd output_val()
    {
        return node.value;
    }

    Eigen::MatrixXd compute_gradient() {
        return node.grad;
    }
};



class Variable : public Nodes {
public:
    Variable(Eigen::MatrixXd X) {
        node.value = X;
        node.need_update = true;
    }
    ~Variable() {

    }
    Eigen::MatrixXd output_val()
    {
        return node.value;
    }

    Eigen::MatrixXd compute_gradient() {
        return node.grad;
    }
};


class Add : public Nodes {
public:
    Add(Nodes* left_node, Nodes* right_node) { //left_node: class.root
        last_left = left_node;
        last_right = right_node;
        last_left->next = this;
        last_right->next = this;
    }
    ~Add() {

    }

    Eigen::MatrixXd output_val()
    {
        node.value = Eigen::MatrixXd::Zero(last_left->node.value.rows(), last_left->node.value.cols());
        if ((last_left->node.value.rows() == last_right->node.value.rows()) && (last_left->node.value.cols() == last_right->node.value.cols())) {
            node.value = last_left->node.value + last_right->node.value;
        }
        else if ((last_left->node.value.cols() == last_right->node.value.cols())) {
            for (int i = 0; i <last_left->node.value.rows(); i++) {
                node.value.row(i) = last_left->node.value.row(i) + last_right->node.value;
            }
        }
        return node.value;
    }

    Eigen::MatrixXd compute_gradient() {
        if (last_left->node.require_grad == true) {
            last_left->node.sub_grad = Eigen::MatrixXd::Ones(last_left->node.value.rows(), last_left->node.value.cols());
        }
        if (last_right->node.require_grad == true) {
            last_right->node.sub_grad = Eigen::MatrixXd::Ones(last_left->node.value.rows(), last_left->node.value.cols());  //there is a trick here! the same as above
        }
        
        return node.grad;
    }

};


class Dot : public Nodes {
public:
    Dot(Nodes* left_node, Nodes* right_node) {
        node.dot = true;
        last_left = left_node;
        last_right = right_node;
        last_left->next = this;
        last_right->next = this;
    }
    ~Dot() {

    }
    Eigen::MatrixXd output_val() {
        if (last_left->node.value.cols() == last_right->node.value.rows())
            node.value = last_left->node.value * last_right->node.value;
        return node.value;
    }

    Eigen::MatrixXd compute_gradient() {
        if (last_left->node.require_grad)
            last_left->node.sub_grad = last_right->node.value.transpose();
        if (last_right->node.require_grad)
            last_right->node.sub_grad = last_left->node.value.transpose();
        return node.grad;
    }
};

class Minus : public Nodes {
public:
    Minus(Nodes* left_node, Nodes* right_node) { //left_node: class.root
       last_left = left_node;
       last_right = right_node;
       last_left->next = this;
       last_right->next = this;
    }
    ~Minus() {

    }

    Eigen::MatrixXd output_val()
    {
        node.value = Eigen::MatrixXd::Zero(last_left->node.value.rows(), last_left->node.value.cols());
        if ((last_left->node.value.rows() == last_right->node.value.rows()) && (last_left->node.value.cols() == last_right->node.value.cols())) {
            node.value = last_left->node.value - last_right->node.value;
        }
        else if ((last_left->node.value.cols() == last_right->node.value.cols())) {
            for (int i = 0; i < last_left->node.value.rows(); i++) {
                node.value.row(i) = last_left->node.value.row(i) - last_right->node.value;
            }
        }
        return node.value;
    }

    Eigen::MatrixXd compute_gradient() {
        if (last_left->node.require_grad == true) {
            last_left->node.sub_grad = Eigen::MatrixXd::Ones(last_left->node.value.rows(), last_left->node.value.cols());
        }
        if (last_right->node.require_grad == true) {
            last_right->node.sub_grad = -1 * Eigen::MatrixXd::Ones(last_right->node.value.rows(), last_right->node.value.cols());
        }

        return node.grad;
    }

};

class F_Norm : public Nodes {
public:
    F_Norm(Nodes* left_node) {
        last_left = left_node;
        last_left->next = this;
    }
    ~F_Norm() {

    }
    Eigen::MatrixXd output_val()
    {   
        node.value = Eigen::MatrixXd::Zero(1,1);
        double sum = 0;
        int r = last_left->node.value.rows();
        int c = last_left->node.value.cols();
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                sum += pow(last_left->node.value(i, j), 2);
            }
        }

        node.value << sum / r;
        return node.value;
    }

    Eigen::MatrixXd compute_gradient() {
        if (last_left->node.require_grad)
            last_left->node.sub_grad = last_left->node.value + last_left->node.value;
        return node.grad;
    }
};

//activation function
class Sigmoid : public Nodes {
public:
    Sigmoid(Nodes * left_node) {
        last_left = left_node;
        node.value = last_left->node.value;
        left_node->next = this;
    }

    ~Sigmoid() {

    }

    Eigen::MatrixXd output_val()
    {
        node.value = 1 / (1 + exp(-1 * last_left->node.value.array()));
        return node.value;
    }

    Eigen::MatrixXd compute_gradient() {
        if (last_left->node.require_grad)
            last_left->node.sub_grad = (1 / (1 + exp(-1 * last_left->node.value.array()))).cwiseProduct(1 - 1 / (1 + exp(-1 * last_left->node.value.array())));
        return node.grad;
    }
};

class ReLu : public Nodes {
public:
    ReLu(Nodes* left_node) {
        last_left = left_node;
        node.value = last_left->node.value;
        left_node->next = this;
    }

    ~ReLu() {

    }

    Eigen::MatrixXd output_val()
    {
        node.value = (last_left->node.value.cwiseAbs() + last_left->node.value) / 2;
        return node.value;
    }

    Eigen::MatrixXd compute_gradient() {
        if (last_left->node.require_grad) {
            int row_num = last_left->node.value.rows();
            int col_num = last_left->node.value.cols();
            last_left->node.sub_grad = Eigen::MatrixXd::Zero(row_num, col_num);
            for (int r = 0; r < row_num; r++) {
                for (int c = 0; c < col_num; c++) {
                    if (last_left->node.value(r, c) > 0)
                        last_left->node.sub_grad(r, c) = 1;
                }
            }
        }

        return node.grad;
    }
};

class Tanh : public Nodes {
public:
    double k;
    Tanh(Nodes* left_node) {
        last_left = left_node;
        node.value = last_left->node.value;
        left_node->next = this;
        k = 6;

    }
    ~Tanh() {

    }

    Eigen::MatrixXd output_val()
    {
        node.value = (k * last_left->node.value.array().exp() - (-1 * k * last_left->node.value.array()).exp()) / (k * last_left->node.value.array().exp() + (-1 * k * last_left->node.value.array()).exp() + 1e-10);

        return node.value;
    }

    Eigen::MatrixXd compute_gradient() {
        if (last_left->node.require_grad) {
            last_left->node.sub_grad = k * (1 - ((k * last_left->node.value.array().exp() - (-1 * k * last_left->node.value.array()).exp()) / (k * last_left->node.value.array().exp() + (-1 * k * last_left->node.value.array()).exp() + 1e-10)).cwiseAbs2());
        }

        return node.grad;
    }

};

class Lee_Osc : public Nodes {
public:
    double Lee[1001][100], s;
    Lee_Osc(Nodes * left_node){
        last_left = left_node;
        node.value = last_left->node.value;
        left_node->next = this;
        s = 4;
        get_table(Lee);

    }
    ~Lee_Osc(){

    }
    void get_table(double Lee[1001][100]) {
        int N = 600;         // n = no.of time step default is 1000
        //parameter for tanh function
        int a1 = 5, a2 = 5, b1 = 1, b2 = 1, eu = 0, ev = 0, c = 1;      // default is 5
;       // default is 5
;       //u threshold default is 0
;       //v threshold defalut is 0
;       //Decay constant
        double k = 500, e = 0.02;
        int x = 0;        //x index of Lee()
        double u = 0.2, v = 0, w = 0, z = 0.2;
        //double Lee[1001][100];
        double tempu, tempv, sim, sign;
        for (double i = -1; i <= 1.001; i += 0.002) {
            sign = (i == 0) ? 0 : ((i > 0) ? 1 : -1);
            sim = i + 0.02 * sign;
            for (int t = 0; t < N - 1; t++) {
                tempu = a1 * u - a2 * v + sim - eu;
                tempv = b1 * u - b2 * v - ev;
                u = (exp(s * tempu) - exp((-1) * s * tempu)) / (exp(s * tempu) + exp((-1) * s * tempu));
                v = (exp(s * tempv) - exp((-1) * s * tempv)) / (exp(s * tempv) + exp((-1) * s * tempv));
                w = (exp(s * sim) - exp((-1) * s * sim)) / (exp(s * sim) + exp((-1) * s * sim));
                z = ((u - v) * exp(-1 * k * sim * sim) + c * w);
                if (t >= 499) {
                    Lee[x][t - 499] = z;
                }
            }
            x += 1;
        }

    }

    Eigen::MatrixXd output_val()
    {
        int r = last_left->node.value.rows(), c = last_left->node.value.cols();
        node.value = Eigen::MatrixXd::Zero(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                if (last_left->node.value(i, j) > 1) {
                    node.value(i, j) = 1;
                }
                else if (last_left->node.value(i, j) < -1) {
                    node.value(i, j) = -1;
                }
                else {
                    int row_ind = floor((last_left->node.value(i, j) - (-1)) / 0.002) + 0;
                    int col_ind = (rand() % (99 - 0 + 1)) + 0;
                    node.value(i, j) = Lee[row_ind][col_ind];
                }
                    
            }
        }
        
        return node.value;
    }

    Eigen::MatrixXd compute_gradient() {
        if (last_left->node.require_grad) {
            
            last_left->node.sub_grad = s * (1 - ((s * last_left->node.value.array().exp() - (-1 * s * last_left->node.value.array()).exp()) / (s * last_left->node.value.array().exp() + (-1 * s * last_left->node.value.array()).exp() + 1e-10)).cwiseAbs2());
        }
            
        return node.grad;
    }

};