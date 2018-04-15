using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    /// <summary>
    /// A simple class to convey training telemetry from NeuralNetwork
    /// </summary>
    public class TrainingTelemetry
    {
        public int Iteration { get; set; }
        public double[][,] Weights { get; set; }
        public double[][] Bias { get; set; }
        public double[] Error { get; set; }

    }
}
