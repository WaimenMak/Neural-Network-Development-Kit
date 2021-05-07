# Neural-Network-Development-Kit
A Neural Network Development Kit for MT4. The main file contains the computation graph algorithm. The Dll1 file is the api of this project, it contains the Dll1.dll file (maxnet.dll). Project1 is for debugging.

## qfsdk.mqh

The qffc.mq4 file is just a demo example that imitates the program in the DemoFann.pdf. Of course, you can use your own data preprocessing methods.

Note that only the maxnet.dll (i.e. the Dll1.dll) file should be imported in the qfsdk.mqh. The other DLL files are required but won't be used during the MQL programming. If you have Visual Studio on your computer, it is fine to delete the other DLL files except for the maxnet.dll.  Happy programming, thanks!

### API

#### build_net()

```C
bool build_net(int& nn_layer, int& feature_num, int& nn_output, const double &inputVector[], const double  &outputVector[], const int &neuron_num[], const int & active[]);
```

`nn_layer` is the number of the hidden layers, `feature_num` is the feature number of the training sample. `nn_output` is the dimension of the label.

`inputVector`: training sample

`outputVector`: label

`neuron_num`: an int type array, stores the neuron number of each hidden layer

`activation`: the array that indicates which activation function to be used. (0: ReLu  ; 1: Sigmoid ; 2: Tanh ; 3: Lee_oscillator )

`return` build the network successfully or not.

```C
int         iter = 5;                    //training iteration for each batch
double      learn_rate = 0.01;

double targetMSE = 0.002;
int feature_num = 3;
int opt  =  0;                  //optmizer : 1 for Adagrad, else gradient descent
int neuron_num[] = {10,3};      //hidden_layer neuron num

//0:ReLu ; 1:Sigmoid ; 2:Tanh ; 3:Lee_oscillator 
int activation[] = {0,0};

bool s = build_net(hidden_layer, feature_num, nn_output, inpt, oupt, neuron_num, activation);
```

#### Predict()

```C
void Predict(int feature_num, int nn_output, const double &inputVector[], double & y_hat[]);
```

`y_hat`: an array that stores the prediction, it should be initialized before the function is called.

```C
double p_hat[] = {0};  //output_num, initialize
Predict(feature_num, nn_output, inputVector, p_hat);
```

#### train()

```C
double train(int feature_num, int nn_output, const double & inputVector[], const double & outputVector[], int iteration, const double& lr, const int& opt);
```

`iteration`: for each sample, we need to update the weights few time, this parameter controls the number of update.

`lr`: the learning rate.

`opt`: the choice for the optimizer, recommend to use the gradient descent, Adaptive gradient descent is not yet mature.

`return`: the mean square error.

#### release_net()

```C
void release_net();
```

release the memory of the network, it should be called if the network wouldn't be used anymore. Also we need this when we want to reconstruct the network.

### Parameter settings

This is an example for the parameter setting.

```c
int         nn_layer   = 4;               // Number of layers (4 including input and output layers)
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

//0:ReLu ; 1:Sigmoid ; 2:Tanh ; 3:Lee_oscillator 
int activation[] = {0,0};
```



