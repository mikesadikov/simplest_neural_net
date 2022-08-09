
## The simplest neural network in C++

A simple neural network has been implemented, consisting of four input neurons and one output neuron.

[Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) is used as an activation function, the output can take values ​​from 0 to 1. 
The initial weights (weight matrix) have a value of 0.5.

As an example for training, the dataset [Fischer's Irises](https://en.wikipedia.org/wiki/Iris_flower_data_set) is used.
In this project, for simplicity, two types of irises out of three available in the dataset are used - only setosa and versicolor.

The program input is a CSV file with data for training, it looks like this:
```
4.9,3.1,1.5,0.1,setosa
5.9,3.0,4.2,1.5,versicolor
5.0,3.2,1.2,0.2,setosa
6.0,3.4,4.5,1.6,versicolor
6.8,2.8,4.8,1.4,versicolor
etc.
```
Four numbers are some parameters of irises in centimeters, setosa and versicolor are types of irises: bristly iris (Iris setosa) and multi-colored iris (Iris versicolor). 
These four numbers are fed into our input layer, which consists of four neurons.

After training, the input of the neural network is a test data set taken from the same dataset, but not used in training.
To display the neural network metrics, I used a third-party project [accuracy-evaluation-cpp](https://github.com/ashokpant/accuracy-evaluation-cpp)

The output of the program looks like this:
```
Train...
Epoch 1:
Accuracy Evaluation Results
=======================================
	Average System Accuracy(%): 49.3421
	System Error(%)           : 50.6579
	Precision (Micro)(%)      : 49.3421
	Recall (Micro)(%)         : 49.3421
	Fscore (Micro)(%)         : 49.3421
	Precision (Macro)(%)      : 49.3421
	Recall (Macro)(%)         : 49.342
	Fscore (Macro)(%)         : 49.342
Epoch 2:
Accuracy Evaluation Results
=======================================
	Average System Accuracy(%): 48.75
	System Error(%)           : 51.25
	Precision (Micro)(%)      : 48.75
	Recall (Micro)(%)         : 48.75
	Fscore (Micro)(%)         : 48.75
	Precision (Macro)(%)      : -nan
	Recall (Macro)(%)         : 50
	Fscore (Macro)(%)         : -nan
Epoch 3:
Accuracy Evaluation Results
=======================================
	Average System Accuracy(%): 56.25
	System Error(%)           : 43.75
	Precision (Micro)(%)      : 56.25
	Recall (Micro)(%)         : 56.25
	Fscore (Micro)(%)         : 56.25
	Precision (Macro)(%)      : 56.25
	Recall (Macro)(%)         : 56.3492
	Fscore (Macro)(%)         : 56.2996
Epoch 4:
Accuracy Evaluation Results
=======================================
	Average System Accuracy(%): 87.1626
	System Error(%)           : 12.8374
	Precision (Micro)(%)      : 87.1626
	Recall (Micro)(%)         : 87.1626
	Fscore (Micro)(%)         : 87.1626
	Precision (Macro)(%)      : 87.1626
	Recall (Macro)(%)         : 88.5886
	Fscore (Macro)(%)         : 87.8698
Epoch 5:
Accuracy Evaluation Results
=======================================
	Average System Accuracy(%): 100
	System Error(%)           : 0
	Precision (Micro)(%)      : 100
	Recall (Micro)(%)         : 100
	Fscore (Micro)(%)         : 100
	Precision (Macro)(%)      : 100
	Recall (Macro)(%)         : 100
	Fscore (Macro)(%)         : 100

************* Prediction: *************
Test dataset:
0 0 0 0 1 1 1 1 
Prediction result:
0 0 0 0 1 1 1 1 
```

As you can see, only 5 epochs of training were enough!

## Build instructions:

To build the project, run the following commands:
```
mkdir build && cd build
cmake ..
cmake --build .
```

