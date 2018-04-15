using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    class Program
    {
        static void Main(string[] args)
        {
            //Simple Artificial Neural Network
            //We will construct a 3 layer ANN to learn the XOR function.

            //Define the network layers
            //layer 0 is the input layer, 1 is the hidden layer, 2 is the output layer.
            int[] layers = new[] { 2, 2, 1 };
            var nn = new NeuralNetwork(layers)
            {
                Iterations = 1000,              //training iterations
                Alpha = 3.5,                    //learning rate, lower is slower, too high may not converge.
                L2_Regularization = true,       //set L2 regularization to prevent overfitting
                Lambda = 0.0003,                //strength of L2
                Rnd = new Random(12345)         //provide a seed for repeatable outputs
            };

            //Define the inputs and the outputs.
            //The last value in these training sets is the expected output.

            //Simplest, learn that the first input value is always the answer
            //var training = new double[][]
            //{
            //    new double[]{ 1, 1, 1 },
            //    new double[]{ 0, 1, 0 },
            //    new double[]{ 1, 0, 1 },
            //    new double[]{ 0, 1, 0 },
            //    new double[]{ 1, 1, 1 },
            //    new double[]{ 0, 0, 0 },
            //};

            //Simple AND
            //var training = new double[][]
            //{
            //    new double[]{ 1, 1, 1 },
            //    new double[]{ 0, 1, 0 },
            //    new double[]{ 1, 0, 0 },
            //    new double[]{ 0, 1, 0 },
            //    new double[]{ 1, 1, 1 },
            //    new double[]{ 0, 0, 0 },
            //};

            //Simple OR
            //var training = new double[][]
            //{
            //    new double[]{ 1, 0, 1 },
            //    new double[]{ 0, 1, 1 },
            //    new double[]{ 1, 1, 1 },
            //    new double[]{ 0, 0, 0 },
            //    new double[]{ 1, 1, 1 },
            //    new double[]{ 0, 0, 0 },
            //    new double[]{ 1, 0, 1 },
            //    new double[]{ 0, 1, 1 },
            //};

            //Simple XOR
            var training = new double[][]
            {
                new double[]{ 1, 0, 1 },
                new double[]{ 0, 1, 1 },
                new double[]{ 1, 1, 0 },
                new double[]{ 0, 0, 0 },
                new double[]{ 1, 1, 0 },
                new double[]{ 0, 0, 0 },
                new double[]{ 1, 0, 1 },
                new double[]{ 0, 1, 1 },
            };


            //Normalize the input to -1,+1 ?
            //We want a 0 mean, and 1 stdev.


            //Take the first 2 columns as input, and last 1 column as target y (the expected label)
            var input = new double[training.GetLength(0)][];
            for (int i = 0; i < training.GetLength(0); i++)
            {
                input[i] = new double[layers[0]];
                for (int j = 0; j < layers[0]; j++)
                    input[i][j] = training[i][j];
            }
                
            //Create the expected label array
            var y = new double[training.GetLength(0)];
            for (int i = 0; i < training.GetLength(0); i++)
                y[i] = training[i][layers[0]];


            //Let's also monitor training by providing a delegate function
            nn.Monitor = delegate(TrainingTelemetry t)
            {
                Console.CursorLeft = 0;
                Console.CursorTop = 0;

                //Display some information about its learning at each iteration
                Console.WriteLine($"Iteration {t.Iteration}");

                //Display some sample data
                Console.WriteLine($"{nn.Predict(new[] { 0.0, 0.0 })[0]} -> 0");
                Console.WriteLine($"{nn.Predict(new[] { 0.0, 1.0 })[0]} -> 1");
                Console.WriteLine($"{nn.Predict(new[] { 1.0, 0.0 })[0]} -> 1");
                Console.WriteLine($"{nn.Predict(new[] { 1.0, 1.0 })[0]} -> 0");

                //Just for fun lets print out the weights and biases
                Console.WriteLine("\nWeights:");
                for (int l = 0; l < t.Weights.Length; l++)
                {
                    Console.WriteLine($"  Layer {l}");
                    Console.WriteLine("  --------------------------");
                    for (int j = 0; j < t.Weights[l].GetLength(0); j++)
                    {
                        for (int k = 0; k < t.Weights[l].GetLength(1); k++)
                            Console.Write("  {0:#.##}\t", t.Weights[l][j, k]);

                        Console.WriteLine();
                    }
                }

                Console.WriteLine("\nBiases:");
                Console.WriteLine("--------------------------");
                for (int l = 1; l < t.Bias.Length; l++)
                {
                    for (int n = 0; n < t.Bias[l].Length; n++)
                        Console.Write("  {0:#.##}\t", t.Bias[l][n]);
                    Console.WriteLine();
                }

                //Display average error
                var absCost = (double)t.Error.Sum(v => Math.Abs(v)) / t.Error.Length;
                Console.WriteLine("\nError {0:#.#####}", absCost);

                //You can step through manually if you wish
                //Console.ReadKey(true);
            };


            //Begin training the network to learn the function that matches our data.
            nn.Train(input, y);

            //Confirm that its worked
            Console.WriteLine($"The network learned XOR(1,0)={nn.Predict(new[] { 1.0, 0.0 })[0]}");
            Console.WriteLine($"The network learned XOR(1,1)={nn.Predict(new[] { 1.0, 1.0 })[0]}");
            Console.WriteLine("press any key to continue");
            Console.ReadKey(true);
        }
    }
}
