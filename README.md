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


The bias and the weights on the inputs are learned parameters which are not initially defined.  
The choice of activation function is usually a parameter of the model, but we will discuss this later.
So mathematically each input would be multiplied by the weight of that connection, and summed with all the other weighted inputs, and offset by a bias, which cana be written as:

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?y&space;=&space;\sigma&space;(\sum_{i}&space;x_i&space;w_i&space;&plus;&space;b)" title="a = \sigma (\sum_{i} x_i w_i + b)" />
</p>
where sigma is the activation function, of the sum of the weighted inputs plus the bias.



There are several activation functions which may be used, each having various advantages.  It also must be differentiable, and there are many options, but we will touch on that later.




Most importantly 
It then runs through the activation function and produces an output y as shown below




![alt text](https://i.stack.imgur.com/9jzpy.jpg "Neural Network Diagram")

# Further Resources
