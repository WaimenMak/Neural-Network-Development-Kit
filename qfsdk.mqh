//+------------------------------------------------------------------+
//|                                                        qfsdk.mqh |
//|                                                      Weiming Mai |
//|                                     https://github.com/WaimenMak |
//+------------------------------------------------------------------+
#property copyright "Weiming Mai"
#property link      "https://github.com/WaimenMak"
#property strict

#import "D:\下载\Quant\Proj\Dll1\Debug\Dll1.dll"
//#import "maxnet.dll"
int myAdd(int a, int b);
bool build_net(int& nn_layer, int& feature_num, int& nn_output, const double &inputVector[], const double  &outputVector[], const int &neuron_num[], const int & active[]);
void Predict(int feature_num, int nn_output, const double & inputVector[], double & y_hat[]);
double train(int feature_num, int nn_output, const double & inputVector[], const double & outputVector[], int iteration, const double& lr, const int& opt);
void release_net();

#import
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
// #define MacrosHello   "Hello, world!"
// #define MacrosYear    2010
//+------------------------------------------------------------------+
//| DLL imports                                                      |
//+------------------------------------------------------------------+
// #import "user32.dll"
//   int      SendMessageA(int hWnd,int Msg,int wParam,int lParam);
// #import "my_expert.dll"
//   int      ExpertRecalculate(int wParam,int lParam);
// #import
//+------------------------------------------------------------------+
//| EX5 imports                                                      |
//+------------------------------------------------------------------+
// #import "stdlib.ex5"
//   string ErrorDescription(int error_code);
// #import
//+------------------------------------------------------------------+
