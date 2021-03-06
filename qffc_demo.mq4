//+------------------------------------------------------------------+
//|                                                           WM.mq4 |
//|                                                      Weiming Mai |
//|                                     https://github.com/WaimenMak |
//+------------------------------------------------------------------+
#property copyright "Weiming Mai"
#property link      "https://github.com/WaimenMak"
#property version   "1.00"
#property strict
#include "qfsdk.mqh"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

int         nn_layer   = 4;               // Number of layers (4 including input and output layers)
//int         nn_input   = 20;              // Number of input neurones, 5-Day time series = 20 neurons
//int         nn_hidden1 = 10;              // Number of neurones on the first hidden layer
//int         nn_hidden2 = 10;              // number on the second hidden layer
int         nn_output  = 1;               // number of outputs
double      trainingData[][4];           // IMPORTANT! size = nn_input + nn_output
int         maxTraining = 1000;  
int         hidden_layer = nn_layer - 2;

int         iter = 5;                    //training iteration
double      learn_rate = 0.01;

double targetMSE = 0.002;
int feature_num = 3;
int opt  =  0;                  //optmizer
int neuron_num[] = {10,3};      //hidden_layer neuron num

//0:ReLu ; 1:Sigmoid ; 2:Tanh ; 3:Lee_oscillator ; default: Line
int activation[] = {0,2};

int OnInit()
  {
//---
   int i;
   double MSE;
//---
   ArrayResize(trainingData,1);
   prepareData("train",1,2,3,1);
   prepareData("train",8,12,20,1);
   prepareData("train",4,6,8,1);
   prepareData("train",0,5,11,1);
   // UP DOWN = DOWN / if a < b && b > c then output = 0
   prepareData("train",1,2,1,0);
   prepareData("train",8,10,7,0);
   prepareData("train",7,10,7,0);
   prepareData("train",2,3,1,0);
   // DOWN DOWN = DOWN / if a > b && b > c then output = 0
   prepareData("train",8,7,6,0);
   prepareData("train",20,10,1,0);
   prepareData("train",3,2,1,0);
   prepareData("train",9,4,3,0);
   prepareData("train",7,6,5,0);
   // DOWN UP = UP / if a > b && b < c then output = 1
   prepareData("train",5,4,5,1);
   prepareData("train",2,1,6,1);
   prepareData("train",20,12,18,1);
   prepareData("train",8,2,10,1);
   
   double inpt[3] = {1,2,3};
   double oupt[] = {1};
   bool s = build_net(hidden_layer, feature_num, nn_output, inpt, oupt, neuron_num, activation);
   /*
   double p = 0;
   double p_hat[] = {0};
   if (s){
      Predict(feature_num, nn_output, inpt, p_hat);
   }
   */
//   release_net();
   
//   build_net(hidden_layer, feature_num,nn_output, inpt, oupt, neuron_num);   
   Print("##### INIT #####");
   printDataArray();
   int a = myAdd(3, 4);
   Print(a);
   Print(s);
   //Print(p_hat[0]);
   Print("##### finish #####");
   
   
   for (i=0;i<maxTraining;i++) {    //epoch
      MSE = teach(); // everytime the loop run, the teach() function is activated.Check the comments associated to this function to understand more.
      if (MSE < targetMSE) { // if the MSE is lower than what we defined (here
         debug("training finished. Trainings ",i+1); // then we print in the console how many training it took them to understand
         i = maxTraining; // and we go out of this loop
         }
      }
      debug("MSE",MSE);
//---
   Print("##### RUNNING #####");
   debug("1,2,3 = UP DOWN = DOWN. Should output 1.","");
   prepareData("compute",1,2,3,0);
   debug("1,3,1 = UP DOWN = DOWN. Should output 0.","");
   prepareData("compute",1,3,1,0);
   debug("1,2,3 = UP UP = UP. Should output 1.","");
   prepareData("compute",1,2,3,0);
   debug("3,2,1 = DOWN DOWN = DOWN. Should output 0.","");
   prepareData("compute",3,2,1,0);
   debug("45,2,89 = DOWN UP = UP. Should output 1.","");
   prepareData("compute",45,2,89,0);
   debug("1,3,23 = UP UP = UP. Should output 1.","");
   prepareData("compute",1,3,23,0);
   debug("7,5,6 = DOWN UP = UP. Should output 1.","");
   prepareData("compute",7,5,6,0);
   debug("2,8,9 = UP UP = UP. Should output 1.","");
   prepareData("compute",2,8,9,0);
   
   Print("=================================== END EXECUTION================================");
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
     release_net();

  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
  }
//+------------------------------------------------------------------+


void prepareData(string action, double a, double b, double c, double output) {
   double inputVector[];
   double outputVector[];
   // we resize the arrays to the right size
   ArrayResize(inputVector,feature_num);
   ArrayResize(outputVector,nn_output);
   inputVector[0] = a;
   inputVector[1] = b;
   inputVector[2] = c;
   outputVector[0] = output;
   if (action == "train") {
      addTrainingData(inputVector,outputVector);
   }
   if (action == "compute") {
      compute(inputVector);
   }
// if you have more input than 3, just change the structure of this function.
}

void addTrainingData(double &inputArray[], double &outputArray[]) {
   int j;
   int size = ArraySize(trainingData);
   int bufferSize = ArraySize(trainingData)/(feature_num+nn_output)-1;
   //register the input data to the main array
   for (j=0;j<feature_num;j++) {
      trainingData[bufferSize][j] = inputArray[j];
   }
   for (j=0;j<nn_output;j++) {
      trainingData[bufferSize][feature_num+j] = outputArray[j];
   }
   ArrayResize(trainingData,bufferSize+2);
}

void printDataArray() {
   int i,j;
   int bufferSize = ArraySize(trainingData)/(feature_num+nn_output)-1;
   string lineBuffer = "";
   for (i=0;i<bufferSize;i++) {
      for (j=0;j<(feature_num+nn_output);j++) {
         lineBuffer = StringConcatenate(lineBuffer, trainingData[i][j], ",");
      }
      debug("DataArray["+i+"]", lineBuffer);
      lineBuffer = "";
   }
}

void debug(string a, string b) {
   Print(a+" ==> "+b);
}




double teach() {
   int i,j;
   double batch_MSE = 0;
   double inputVector[];
   double outputVector[];
   ArrayResize(inputVector,feature_num);
   ArrayResize(outputVector,nn_output);
   int call;
   int bufferSize = ArraySize(trainingData)/(feature_num+nn_output)-1;
   for (i=0;i<bufferSize;i++) {
      for (j=0;j<feature_num;j++) {
         inputVector[j] = trainingData[i][j];
      }
      outputVector[0] = trainingData[i][3];
      //f2M_train() is showing the neurones only one example at a time.
      //call = f2M_train(ann, inputVector, outputVector);
      batch_MSE += train(feature_num,nn_output,inputVector, outputVector, iter, learn_rate, opt)/bufferSize;
      
      
   }
   // Once we have show them an example, we check if how good they are by checking their MSE. 
   //MSE = f2M_get_MSE(ann);
   return(batch_MSE);
}

double compute(double &inputVector[]) {
//int j;
   int out;
   double output;
   ArrayResize(inputVector,feature_num);
   // We sent new data to the neurones.
   //out = f2M_run(ann, inputVector);
   double p_hat[] = {1};  //output_num
   Predict(feature_num, nn_output, inputVector, p_hat);
   // and check what they say about it using f2M_get_output().
   //output = f2M_get_output(ann, 0);
   output = p_hat[0];
   debug("Computing()",output);
   return(output);
   }
