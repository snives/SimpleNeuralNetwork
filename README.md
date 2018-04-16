# SimpleNeuralNetwork
Simple artificial neural network implemented in C#

## Summary


## Background
Artificial Neural Networks (ANNs) are a very powerful tool within the supervised learning category of machine learning.
Given some labeled training data they can learn an objective function to predict the optimal output for a given input.
The concept for this algorithm is patterned after the operation of biological neurons, 
where a sequence of impulses travel from neuron to neuron, in various graph like pathways, and at each junction the signal is selectively inhibited or passed along.

Each neuron makes this decision by whether the combined inputs passes some activation threshold, if so, it passes along this signal. With proper training, a network of neurons will learn the correct behavior for a given set of inputs and will ultimately produce the correct response.

## Modeling a single neuron
Moving from a biological representation to a mathematical representation we can model an individual neuron as a black box function which takes multiple inputs, does something, and produces an output.  A neuron should have several inputs carrying a signal into it, each with a different weight.  The inputs with the strongest weights should have the greatest influence on the neuron.  The threshold will be modeled by a bias which the combined inputs must overcome to activate the neuron.  The activation of the neuron will then be modeled by an activation function, the result of which will produce the output.

<p align="center">
  <img src="https://i.stack.imgur.com/VqOpE.jpg" /><br>
Figure a.
</p>


The bias and the weights on the inputs are learned parameters.
The choice of activation function is usually a parameter of the model, discussed below.  Mathematically each input would be multiplied by the weight of that connection, and summed with all the other weighted inputs, offset by a bias, and ran through an activation function, which can be written as

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y&space;=&space;\sigma&space;(\sum_{i}&space;x_i&space;w_i&space;&plus;&space;b)" title="a = \sigma (\sum_{i} x_i w_i + b)" />
</p>

where sigma is the activation function, i is the index of each input, x is the input, w is the weight of the connection, and b is the bias.





## Activation Functions
There are several activation functions which may be used, each having different characteristics, but for this demonstration we will be using the sigmoid, but we will cover the most popular below.

### Linear
The linear activation is the simplest function, however because it is a linear function it is not able to solve non-linear functions, and is not bounded.

<img src="https://latex.codecogs.com/gif.latex?\sigma&space;(x)&space;=&space;x" title="\sigma (x) = x" />

<img src="https://github.com/snives/SimpleNeuralNetwork/blob/master/Docs/desmos-graph-linear.png?raw=true" height="100px" style="border: 1px solid #ccc;margin-left:20px;"/>

### Sigmoid
The sigmoid, also known as the logistic function, is non-linear with bounds of [0,1].  Its continuous, differentiable and squashes extreme values to 0 or 1.

<img src="https://latex.codecogs.com/gif.latex?\sigma&space;(x)&space;=&space;\frac{1}{1\&space;&plus;\exp\left(-x\right)}" title="\sigma (x) = \frac{1}{1\ +\exp\left(-x\right)}" />

<img src="https://github.com/snives/SimpleNeuralNetwork/blob/master/Docs/desmos-graph-sigmoid.png?raw=true" height="100px" style="border: 1px solid #ccc;margin-left:20px;"/>

### Hyperbolic Tangent
The TanH function is also non-linear with limits at [-1,1].  It's continuous, differentiable, and squashes extreme values to -1 or 1.  Note its derivative near zero is greater than the sigmoid which if the pre-activation value is relatively near zero then in practice it should converge faster.  This also may suffer from the vanishing gradient problem as the derivative of this function at anywhere outside [-2,2] are near zero.  

<img src="https://latex.codecogs.com/gif.latex?\sigma&space;(x)&space;=&space;\frac{\exp\left(x\right)-\exp\left(-x\right)}{\exp\left(x\right)&plus;\exp\left(-x\right)}" title="\sigma (x) = \frac{\exp\left(x\right)-\exp\left(-x\right)}{\exp\left(x\right)+\exp\left(-x\right)}" />

<img src="https://github.com/snives/SimpleNeuralNetwork/blob/master/Docs/desmos-graph-tanh.png?raw=true" height="100px" style="border: 1px solid #ccc;margin-left:20px;"/>



### ReLU
The rectified linear unit is also a non-linear function however it is very simple.  Its semi-bounded at [0, +inf] but has been shown to work well in deep neural networks because it avoids the vanishing gradient problem.

<img src="https://latex.codecogs.com/gif.latex?\sigma&space;(x)&space;=&space;\max\left(0,x\right)" title="\sigma (x) = \max\left(0,x\right)" />

<img src="https://github.com/snives/SimpleNeuralNetwork/blob/master/Docs/desmos-graph-ReLU.png?raw=true" height="100px" style="border: 1px solid #ccc;margin-left:20px;"/>


## Network of Neurons
The power of neural networks can be seen when we link neuronal units together, sometimes called nodes or units, into a multi-layer neural network (historically referred to as a multi-layer perceptron.)

Our multidimensional input vector X, in this diagram is 5 dimensional.  And it contains a full-mesh network with the hidden layer neurons, which are modeled by our weights.  Given N inputs and M hidden layers, you will have an NxM number of weights.  The same is true for the hidden layer to the output layer.  Networks may have a different number of neurons in each layer, and/or a different number of layers.  This is a free parameter of your model and suitable values will need to be found.

![alt text](https://i.stack.imgur.com/9jzpy.jpg "Neural Network Diagram")

# Further Resources
