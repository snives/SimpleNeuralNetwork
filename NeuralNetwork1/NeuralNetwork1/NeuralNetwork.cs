using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    /// <summary>
    /// A simple artificial neural network using the sigmoid activation function.
    /// </summary>
    public class NeuralNetwork
    {
        /// <summary>
        /// L layers * N neurons
        /// </summary>
        private Neuron[][] Neurons;

        /// <summary>
        /// L-1 layers * from node * to node weights
        /// </summary>
        private double[][,] Weights;

        /// <summary>
        /// The number of learning iterations to perform.
        /// </summary>
        public int Iterations { get; set; } = 5000;

        /// <summary>
        /// Switch to control the application of L2 regularization on learned weights
        /// </summary>
        public bool L2_Regularization { get; set; } = true;

        /// <summary>
        /// The strength of the L2 regularization
        /// </summary>
        public double Lambda { get; set; } = 0.00003;

        /// <summary>
        /// Controls the learning rate.  Increasing for larger jumps but too high may prevent convergence.
        /// </summary>
        public double Alpha { get; set; } = 5.5;

        private int LastLayer;

        public Random Rnd { get; set; } = new Random();

        /// <summary>
        /// An optional delegate function to monitor the training
        /// </summary>
        public Action<TrainingTelemetry> Monitor { get; set; }



        /// <summary>
        /// Construct a new artificial neural network
        /// </summary>
        /// <param name="layers"></param>
        public NeuralNetwork(int[] layers)
        {
            LastLayer = layers.Length - 1;

            //All nodes in the network will be neurons
            Neurons = new Neuron[layers.Length][];
            for (int l = 0; l < layers.Length; l++)
            {
                Neurons[l] = new Neuron[layers[l]];
                //Initialize each layers nodes
                for (int n = 0; n < layers[l]; n++)
                    Neurons[l][n] = new Neuron();
            }
        }

        /// <summary>
        /// Initialize weights randomly*
        /// </summary>
        /// <param name="rnd"></param>
        private void InitializeWeights()
        {
            //It's recommended to initialize weights randomly, with a mean of 0.
            //The variance of these weights should also decrease as you iterate towards
            //the output neurons.  The idea is that we don't want the shallower weights
            //to learn faster than the deeper layers.

            //The intuition is that the complexity of the network is determined by the number of
            //neurons and neurons with zero weights effectively disappear.  The weights distribute
            //corrections to the gradient and if they are large they will effectively learn faster
            //than other neurons.

            //Matrix size is l(i)*l(i+1)
            int layers = Neurons.GetLength(0);
            Weights = new double[layers - 1][,];
            for (int l = 0; l < layers - 1; l++)
            {
                Weights[l] = new double[Neurons[l].Length, Neurons[l + 1].Length];  //2x3, 3x1
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    for (int j = 0; j < Neurons[l + 1].Length; j++)
                        Weights[l][n, j] = WeightFunction(l+1);
                }
            }
        }

        /// <summary>
        /// Sample from a theoretically optimal random distribution of neuron weights
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="rnd"></param>
        /// <returns></returns>
        private double WeightFunction(int layer)
        {
            //Randomly sample a uniform distribution in the range [-b,b] where b is:
            //where fanIn is the number of input units in the weights and
            //fanOut is the number of output units in thes weights
            var fanIn = (layer > 0) ? Neurons[layer - 1].Length : 0;
            var fanOut = Neurons[layer].Length;
            var b = Math.Sqrt(6) / Math.Sqrt(fanIn + fanOut);
            return Rnd.NextDouble() * 2 * b - b;

            //See for more
            //https://keras.io/initializers/#glorot_uniform
            //http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            //https://intoli.com/blog/neural-network-initialization/
            //https://cs231n.github.io/neural-networks-2/#init
        }

        /// <summary>
        /// Initialize biases to zero
        /// </summary>
        private void InitializeBiases()
        {
            int layers = Neurons.GetLength(0);
            for (int l = 1; l < layers; l++)
            {
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    Neurons[l][n].bias = 0.0;
                }
            }
        }

        /// <summary>
        /// Training allows the network to learn how to predict output y given the inputs.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="y"></param>
        public void Train(double[][] input, double [] y)
        {
            InitializeWeights();
            InitializeBiases();

            int iteration = Iterations;
            //int stopEarlyVotes = 0;

            var cost = new double[Neurons[LastLayer].Length];
            
            while (iteration-- > 0)
            {
                //Loop through each input, and compute the prediction
                for (int i = 0; i < input.GetLength(0); i++)
                {

                    //Create perturbed inputs
                    //var perturbed = new double[input[i].Length];
                    //for (int p = 0; p < input[i].Length; p++)
                    //    perturbed[p] = input[i][p] + (rnd.NextDouble() / 10) - 0.05;

                    //FeedForward
                    var output = Predict(input[i]);
                    //var output = Predict(perturbed);
                    
                    //Perform online training by updating the bias and weights after each iteration

                    //Compute the error in the output layer
                    for (int n = 0; n < Neurons[LastLayer].Length; n++)
                    {
                        //Compute the error at the output layer
                        cost[n] = Neurons[LastLayer][n].output - y[i];

                        //Assign the error to the output layer's error
                        Neurons[LastLayer][n].error = (cost[n] * SigmoidPrime(Neurons[LastLayer][n].input + Neurons[LastLayer][n].bias));
                        //But instead of calculating the sigmoid again we can use the output we have to calculate its derivative directly.
                        //Neurons[LastLayer][n].error = (cost[n] * Prime(Neurons[LastLayer][n].output));
                    }

                    //Backpropagate the error through the network
                    BackPropagate();

                    //Adjust the bias by the amount of this error
                    for (int l = 1; l <= LastLayer; l++)
                        for (int n = 0; n < Neurons[l].Length; n++)
                            //Adjust the bias by the derivative of the error (which is the error itself)
                            Neurons[l][n].bias -= (Alpha * Neurons[l][n].error);


                    //Adjust the weights by the amount of this error
                    for (int l = 0; l <= LastLayer - 1; l++)
                    {
                        for (int j = 0; j < Neurons[l].Length; j++)
                            for (int k = 0; k < Neurons[l + 1].Length; k++)
                            {
                                Weights[l][j, k] -= (Alpha * Neurons[l][j].output * Neurons[l + 1][k].error);

                                //Add L2 regularization to prevent overfitting by discouraging high weights
                                if (L2_Regularization)
                                    Weights[l][j, k] -= (Lambda * Weights[l][j, k]);
                            }
                    }
                }
                
                if (Monitor != null)
                {
                    //Produce a jagged array of the bias
                    var bias = new double[Neurons.Length][];
                    for (int i = 0; i < bias.GetLength(0); i++)
                    {
                        bias[i] = new double[Neurons[i].Length];
                        for (int j = 0; j < bias[i].Length; j++)
                            bias[i][j] = Neurons[i][j].bias;
                    }

                    var telemetry = new TrainingTelemetry()
                    {
                        Iteration = iteration,
                        Weights = Weights,
                        Bias = bias,
                        Error = cost
                    };
                    Monitor(telemetry);
                }
                
                //Compute total cost
                //var absCost = (double)cost.Sum(v => Math.Abs(v)) / Neurons[LastLayer].GetLength(0);
                //Console.WriteLine("\nError {0:#.#####}", absCost );

                
                ////Introduce some reweighting to get unstuck from local minimas
                //if (iteration % 100 == 0)
                //{
                //    if (Math.Abs(absCost) > .5)
                //    {
                //        //Random restart since we started in a bad spot
                //        InitializeWeights();
                //        InitializeBiases();
                //        stopEarlyVotes = 0;
                //    }
                //    else if (Math.Abs(absCost) > .025)
                //    {
                //        //zero a weight randomly
                //        //int layer = rnd.Next(0, LastLayer);
                //        //int from = rnd.Next(0, Neurons[layer].GetLength(0));
                //        //int to = rnd.Next(0, Neurons[layer + 1].GetLength(0));
                //        //Console.WriteLine($"Zeroing layer {layer} weight[{from},{to}]");
                //        //Weights[layer][from, to] = 0.0;

                //        //Weights[layer][from, to] = WeightFunction(layer, rnd);
                //    }


                //    //if (math.abs(abscost) < .05)
                //    //{
                //    //    alpha *= 1.5;
                //    //    console.writeline($"alpha: {alpha}");
                //    //}

                //    //Stop early if we have achieved an optimal solution
                //    //Use a windowing function to find the average slope
                //    //once it approaches horizontal then end.
                //    if (Math.Abs(absCost) < .001)
                //    {
                //        stopEarlyVotes++;

                //        if (stopEarlyVotes > 5)
                //            break;
                //    }
                //}
            }
        }



        /// <summary>
        /// Use the neural network to predict the output given some input.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[] Predict(double[] input)
        {
            //Forward propagation
            //Fill the first layers output neurons with input data
            for (int d = 0; d < Neurons[0].Length; d++)
                Neurons[0][d].output = input[d];

            //Feed forward phase
            for (int l = 1; l < Neurons.GetLength(0); l++)
            {
                //Now compute layer l, n is each neuron in layer l
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    //Compute neuron n in layer l
                    double sum = 0;

                    //Iterate over previous layers outputs and weights
                    //j is each of the previous layers neuron
                    for (int j = 0; j < Neurons[l - 1].Length; j++)
                        sum += (Neurons[l - 1][j].output * Weights[l - 1][j, n]);

                    //Store the weighted inputs on input.
                    Neurons[l][n].input = sum;

                    //The output is the sigmoid of the weighted input plus the bias
                    Neurons[l][n].output = Sigmoid(Neurons[l][n].input + Neurons[l][n].bias);
                }
            }

            //prepare a vector of outputs to return
            var outputlayer = Neurons.GetLength(0) - 1;
            var output = new double[Neurons[outputlayer].Length];
            for (int n = 0; n < output.Length; n++)
                output[n] = Neurons[outputlayer][n].output;

            return output;
        }

        /// <summary>
        /// Backpropagate the error proportionately to all neurons by their contribution to the output.
        /// </summary>
        public void BackPropagate()
        {
            //From right to left (output to input)
            for (int l = LastLayer - 1; l > 0; l--)
            {
                for (int n = 0; n < Neurons[l].Length; n++)
                {
                    //matrix form
                    //error in this layer = dot product( weights in next layer,  error in next layer) * sigmaprime(weighted inputs of this layer)

                    //Sum the product of the weight * error in L+1
                    double sum = 0.0;
                    for (int m = 0; m < Neurons[l + 1].Length; m++)
                    {
                        //Weights of L is actually the L + 1 layer
                        sum += (Weights[l][n, m] * Neurons[l + 1][m].error);
                    }

                    Neurons[l][n].error = sum * SigmoidPrime(Neurons[l][n].input + Neurons[l][n].bias);
                    //But instead of calculating the sigmoid again we can use the output we have to calculate its derivative directly.
                    //Neurons[l][n].error = sum * Prime(Neurons[l][n].output);
                }
            }
        }

        /// <summary>
        /// Computes the sigmoid(x) transform function.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        /// <summary>
        /// Computes the derivative of the sigmoid(x)
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        private static double SigmoidPrime(double x)
        {
            return Sigmoid(x) * (1.0 - Sigmoid(x));
        }

        /// <summary>
        /// Compute the derivative of the sigmoid(x) where z=sigmoid(x)
        /// </summary>
        /// <param name="z"></param>
        /// <returns></returns>
        private static double Prime(double z)
        {
            return z * (1.0 - z);
        }

    }
}
