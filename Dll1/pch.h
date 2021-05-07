// pch.h: 这是预编译标头文件。
// 下方列出的文件仅编译一次，提高了将来生成的生成性能。
// 这还将影响 IntelliSense 性能，包括代码完成和许多代码浏览功能。
// 但是，如果此处列出的文件中的任何一个在生成之间有更新，它们全部都将被重新编译。
// 请勿在此处添加要频繁更新的文件，这将使得性能优势无效。

#ifndef PCH_H
#define PCH_H

// 添加要在此处预编译的标头
#include "framework.h"
#include "Nodes2.h"

extern "C" _declspec(dllexport) bool build_net(int& nn_layer, int& feature_num, int& nn_output, const double inputVector[], const double  outputVector[], const int neuron_num[], const int active[]);

extern "C" _declspec(dllexport) void Predict(int feature_num, int nn_output, const double inputVector[], double * y_hat);

extern "C" _declspec(dllexport) int myAdd(int a, int b);

extern "C" _declspec(dllexport) double train(int feature_num, int nn_output, const double inputVector[], const double  outputVector[], int iteration, const double& lr, const int& opt);

extern "C" _declspec(dllexport) void release_net();

#endif //PCH_H
