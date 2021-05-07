//#pragma once
//#include <cstddef>
//#include <Eigen/Dense>              // 稠密矩阵的代数运算
//#include <Eigen/Core>
//#include <math.h>
//
//struct Node {
//    Node* last_right;
//    Node* last_left;
//    Node* next;
//    Eigen::MatrixXd  value;
//    Eigen::MatrixXd grad;
//    Eigen::MatrixXd sub_grad;
//    bool require_grad;
//    bool need_update;
//    bool BN;
//    void* self;
//};
//
//class Nodes {
//public:
//    /*Nodes* last_right;
//    Nodes* last_left;
//    Nodes* next;*/
//    //double* value;
//    /*Eigen::MatrixXd  value;
//    Eigen::MatrixXd grad;
//    Eigen::MatrixXd sub_grad;
//    bool require_grad;
//    bool need_update;
//    bool BN;*/
//    //Node* root;
//    Node  node;
//    Node* root;
//    Nodes(void) {
//        root = &node;
//        node.last_right = NULL;
//        node.last_left = NULL;
//        node.next = NULL;
//        //value = NULL;
//       /* grad = NULL;
//        sub_grad = NULL;*/
//        node.require_grad = true;
//        node.need_update = false;
//        node.BN = false;
//        node.self = this;
//    }
//    ~Nodes(void) {}
//};
//
//class Input : public Nodes {
//public:
//    //Node * root;
//    //template <typename mat>
//    Input(Eigen::MatrixXd  input) {
//        //int y = 1;
//
//        node.value = input;
//    }
//
//    ~Input(void) {}
//
//    Eigen::MatrixXd output_val()
//    {
//        return node.value;
//    }
//
//    Eigen::MatrixXd compute_gradient() {
//
//    }
//        
//};
//
//class Const : public Nodes {
//public:
//    Const(const Eigen::MatrixXd X) {
//        node.value = X;
//        node.require_grad = false;
//    }
//    ~Const() {
//
//    }
//    Eigen::MatrixXd output_val()
//    {
//        return node.value;
//    }
//
//    Eigen::MatrixXd compute_gradient() {
//
//    }
//};
//
//
//
//class Variable : public Nodes {
//public:
//    Variable(Eigen::MatrixXd X) {
//        node.value = X;
//        node.need_update = true;
//    }
//    ~Variable() {
//
//    }
//    Eigen::MatrixXd output_val()
//    {
//        return node.value;
//    }
//
//    Eigen::MatrixXd compute_gradient() {
//
//    }
//};
//
//class Add : public Nodes {
//public:
//    Add(Node * left_node, Node * right_node) { //left_node: class.root
//        node.last_left = left_node;
//        node.last_right = right_node;
//        node.last_left->next = root;
//        node.last_right->next = root;
//    }
//    ~Add() {
//
//    }
//   
//    Eigen::MatrixXd output_val()
//    {
//        node.value = Eigen::MatrixXd::Zero(node.last_left->value.rows(), node.last_left->value.cols());
//        if ((node.last_left->value.rows() == node.last_right->value.rows()) && (node.last_left->value.cols() == node.last_right->value.cols())) {
//            node.value = node.last_left->value + node.last_right->value;
//        }
//        else if ((node.last_left->value.cols() == node.last_right->value.cols())) {
//                for (int i = 0; i < node.last_left->value.rows(); i++) {
//                    node.value.row(i) = node.last_left->value.row(i) + node.last_right->value;
//                }
//        }
//        return node.value;
//    }
//
//    Eigen::MatrixXd compute_gradient() {
//
//    }
//
//};
//
//class Dot : public Nodes {
//public:
//    Dot(Node* left_node, Node* right_node) {
//        node.last_left = left_node;
//        node.last_right = right_node;
//        node.last_left->next = root;
//        node.last_right->next = root;
//    }
//    ~Dot() {
//
//    }
//    Eigen::MatrixXd output_val() {
//        if (node.last_left->value.cols() == node.last_right->value.rows())
//            node.value = node.last_left->value * node.last_right->value;
//        return node.value;
//    }
//
//    Eigen::MatrixXd compute_gradient() {
//
//    }
//};
//
//class Minus : public Nodes {
//public:
//    Minus(Node* left_node, Node* right_node) { //left_node: class.root
//        node.last_left = left_node;
//        node.last_right = right_node;
//        node.last_left->next = root;
//        node.last_right->next = root;
//    }
//    ~Minus() {
//
//    }
//
//    Eigen::MatrixXd output_val()
//    {
//        node.value = Eigen::MatrixXd::Zero(node.last_left->value.rows(), node.last_left->value.cols());
//        if ((node.last_left->value.rows() == node.last_right->value.rows()) && (node.last_left->value.cols() == node.last_right->value.cols())) {
//            node.value = node.last_left->value - node.last_right->value;
//        }
//        else if ((node.last_left->value.cols() == node.last_right->value.cols())) {
//            for (int i = 0; i < node.last_left->value.rows(); i++) {
//                node.value.row(i) = node.last_left->value.row(i) - node.last_right->value;
//            }
//        }
//        return node.value;
//    }
//
//    Eigen::MatrixXd compute_gradient() {
//
//    }
//
//};
//
//class F_Norm : public Nodes {
//public:
//    F_Norm(Node* left_node) {
//        node.last_left = left_node;
//        node.last_left->next = root;
//    }
//    ~F_Norm() {
//
//    }
//    Eigen::MatrixXd output_val()
//    {
//        double sum = 0;
//        for (int i = 0; i < node.last_left->value.rows(); i++) {
//            for (int j = 0; i < node.last_left->value.cols(); j++) {
//                sum += pow(node.last_left->value(i, j),2);
//            }
//        }
//            
//        node.value << sum / (node.last_left->value.rows());
//        return node.value;
//    }
//
//    Eigen::MatrixXd compute_gradient() {
//
//    }
//};