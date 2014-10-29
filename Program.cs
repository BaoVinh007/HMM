using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord;
using Accord.Statistics.Models.Markov;
using Accord.Statistics.Models.Markov.Topology;
using Accord.Statistics.Models.Markov.Learning;
using Accord.Statistics.Distributions.Univariate;

namespace HHM
{
    class Program
    {
        // 1. Compute Probability of random integer sequence
        public static void computeProbRandomIntSeq()
        {
            // Create a hidden Markov model with random parameter probabilities
            HiddenMarkovModel hmm = new HiddenMarkovModel(states: 3, symbols: 2);

            // Create an observation sequence of up to 2 symbols (0 or 1)
            int[] observationSequence = new[] { 0, 1, 1, 0, 0, 1, 1, 1 };

            // Evaluate its log-likelihood. Result is -5.5451774444795623
            double logLikelihood = hmm.Evaluate(observationSequence);
            Console.WriteLine("Log likelihood = " + logLikelihood);

            // Convert to a likelihood: 0.0039062500000
            double likelihood = Math.Exp(logLikelihood);
            Console.WriteLine("Likelihood = " + likelihood);
            Console.Read();
        }

        // 2. A continuous model with Normally distributed emission densities for above example
        public static void computeProbRandomIntSeq2()
        {
            /*
            // Create a hidden Markov model with equal Normal state densities
            NormalDistribution density = new NormalDistribution(mean: 0, stdDev: 1);

            HiddenMarkovModel<normaldistribution> hmm =
                new HiddenMarkovModel<normaldistribution>(states: 3, emissions: density);

            // Create an observation sequence of univariate sequences
            double[] observationSequence = new[] { 0.1, 1.4, 1.2, 0.7, 0.1, 1.6 };

            // Evaluate its log-likelihood. Result is -8.748631199228
            double logLikelihood = hmm.Evaluate(observationSequence);

            // Convert to a likelihood: 0.00015867837561
            double likelihood = Math.Exp(logLikelihood); 
             * */
        }
        
        // 3. Decoding a sequence - CORRECT
        public static void decodingSequence()
        {
            // Create a model with given probabilities
            HiddenMarkovModel hmm = new HiddenMarkovModel(
                transitions: new[,] // matrix A
                {
                    { 0.25, 0.25, 0.00 },
                    { 0.33, 0.33, 0.33 },
                    { 0.90, 0.10, 0.00 },
                },
                emissions: new[,] // matrix B
                {
                    { 0.1, 0.1, 0.8 },
                    { 0.6, 0.2, 0.2 },
                    { 0.9, 0.1, 0.0 },
                },
               initial: new[]  // vector pi
                { 
                    0.25, 0.25, 0.0 
                });

            // Create an observation sequence of up to 2 symbols (0 or 1)
            int[] observationSequence = new[] { 0, 1, 1, 0, 0, 1, 1, 1 };

            // Decode the sequence: the path will be 1-1-1-1-2-0-1
            int[] stateSequence = hmm.Decode(observationSequence);

            for (int i = 0; i < stateSequence.Length; i++)
            {
                Console.Write(stateSequence[i]+"-");
            }
            Console.Read();
        }

        // 4.
        public static void learningDiscreteByBaumWelch()
        {
            // Suppose a set of sequences with something in
            // common. From the example below, it is natural
            // to consider those all start with zero, growing
            // sequentially until reaching the maximum symbol 3

            int[][] inputSequences =
            {
                new[] { 0, 1, 2, 3 },
                new[] { 0, 0, 0, 1, 1, 2, 2, 3, 3 },
                new[] { 0, 0, 1, 2, 2, 2, 3, 3 },
                new[] { 0, 1, 2, 3, 3, 3, 3 },
            };


            // Now we create a hidden Markov model with arbitrary probabilities
            HiddenMarkovModel hmm = new HiddenMarkovModel(states: 4, symbols: 4);

            // Create a Baum-Welch learning algorithm to teach it
            BaumWelchLearning teacher = new BaumWelchLearning(hmm);

            // and call its Run method to start learning
            double error = teacher.Run(inputSequences);

            // Let's now check the probability of some sequences:
            double prob1 = Math.Exp(hmm.Evaluate(new[] { 0, 1, 2, 3 }));       // 0.013294354967987107
            Console.WriteLine("prob 1 = " + prob1);
            double prob2 = Math.Exp(hmm.Evaluate(new[] { 0, 0, 1, 2, 2, 3 })); // 0.002261813011419950
            Console.WriteLine("prob 2 = " + prob2);
            double prob3 = Math.Exp(hmm.Evaluate(new[] { 0, 0, 1, 2, 3, 3 })); // 0.002908045300397080
            Console.WriteLine("prob 3 = " + prob3);
            // Now those obviously violate the form of the training set:
            double prob4 = Math.Exp(hmm.Evaluate(new[] { 3, 2, 1, 0 }));       // 0.000000000000000000
            Console.WriteLine("prob 4 = " + prob4);
            double prob5 = Math.Exp(hmm.Evaluate(new[] { 0, 0, 1, 3, 1, 1 })); // 0.000000000113151816
            Console.WriteLine("prob 5 = " + prob5);
            Console.Read();
        }

        // 5. Computes Forward probabilities for a given hidden Markov model and a set of observations (no scaling). 
        public static void forwardProb()
        {
            // Create the transition matrix A
            double[,] transition =
            {  
                { 0.5, 0.4, 0.1 },
                { 0.1, 0.8, 0.1 },
                { 0.1, 0.6, 0.3 }
            };

            // Create the emission matrix B
            // High-Low
            double[,] emission = 
            {  
                { 0.1, 0.9 },
                { 0.5, 0.5 },
                { 0.9, 0.1 }
            };

            // Create the initial probabilities pi
            double[] initial =
            {
                0.3, 0.4, 0.3
            };

            // Create a new hidden Markov model
            HiddenMarkovModel hmm = new HiddenMarkovModel(transition, emission, initial);

            int[] sequence = new int[] { 0, 0, 0 };

            double logLikelihood;
            double[,] prob = ForwardBackwardAlgorithm.Forward(hmm, sequence,out logLikelihood);
            /*
            for (int i = 0; i < prob.GetLength(0); i++)
            {
                for (int j = 0; j < prob.GetLength(1); j++)
                {
                    Console.Write(prob[i, j] + " ");
                }
                Console.WriteLine();
            }*/
            Console.Write(" Prob = " + logLikelihood);
            Console.Read();
        }

        // 6.Learning Classifier

        public static void learningClassifier()
        {
            // Suppose we would like to learn how to classify the
            // following set of sequences among three class labels: 

            int[][] inputSequences =
            {
                // First class of sequences: starts and
                // ends with zeros, ones in the middle:
                new[] { 0, 1, 1, 1, 0 },        
                new[] { 0, 0, 1, 1, 0, 0 },     
                new[] { 0, 1, 1, 1, 1, 0 },     

                // Second class of sequences: starts with
                // twos and switches to ones until the end.
                new[] { 2, 2, 2, 2, 1, 1, 1, 1, 1 },
                new[] { 2, 2, 1, 2, 1, 1, 1, 1, 1 },
                new[] { 2, 2, 2, 2, 2, 1, 1, 1, 1 },

                // Third class of sequences: can start
                // with any symbols, but ends with three.
                new[] { 0, 0, 1, 1, 3, 3, 3, 3 },
                new[] { 0, 0, 0, 3, 3, 3, 3 },
                new[] { 1, 0, 1, 2, 2, 2, 3, 3 },
                new[] { 1, 1, 2, 3, 3, 3, 3 },
                new[] { 0, 0, 1, 1, 3, 3, 3, 3 },
                new[] { 2, 2, 0, 3, 3, 3, 3 },
                new[] { 1, 0, 1, 2, 3, 3, 3, 3 },
                new[] { 1, 1, 2, 3, 3, 3, 3 },
            };

            // Now consider their respective class labels
            int[] outputLabels =
            {
                /* Sequences  1-3 are from class 0: */ 0, 0, 0,
                /* Sequences  4-6 are from class 1: */ 1, 1, 1,
                /* Sequences 7-14 are from class 2: */ 2, 2, 2, 2, 2, 2, 2, 2
            };


            // We will use a single topology for all inner models, but we 
            // could also have explicitled different topologies for each:

            ITopology forward = new Forward(states: 3);

            // Now we create a hidden Markov classifier with the given topology
            HiddenMarkovClassifier classifier = new HiddenMarkovClassifier(classes: 3,
                topology: forward, symbols: 4);

            // And create a algorithms to teach each of the inner models
            var teacher = new HiddenMarkovClassifierLearning(classifier,

                // We can specify individual training options for each inner model:
                modelIndex => new BaumWelchLearning(classifier.Models[modelIndex])
                {
                    Tolerance = 0.001, // iterate until log-likelihood changes less than 0.001
                    Iterations = 0     // don't place an upper limit on the number of iterations
                });


            // Then let's call its Run method to start learning
            double error = teacher.Run(inputSequences, outputLabels);


            // After training has finished, we can check the 
            // output classificaton label for some sequences. 

            int y1 = classifier.Compute(new[] { 0, 1, 1, 1, 0 });    // output is y1 = 0
            Console.WriteLine("y1 = "+y1);
            int y2 = classifier.Compute(new[] { 0, 0, 1, 1, 0, 0 }); // output is y1 = 0
            Console.WriteLine("y2 = " + y2);
            int y3 = classifier.Compute(new[] { 2, 2, 2, 2, 1, 1 }); // output is y2 = 1
            Console.WriteLine("y3 = " + y3);
            int y4 = classifier.Compute(new[] { 2, 2, 1, 1 });       // output is y2 = 1
            Console.WriteLine("y4 = " + y4);
            int y5 = classifier.Compute(new[] { 0, 0, 1, 3, 3, 3 }); // output is y3 = 2
            Console.WriteLine("y5 = " + y5);
            int y6 = classifier.Compute(new[] { 2, 0, 2, 2, 3, 3 }); // output is y3 = 2
            Console.WriteLine("y6 = " + y6);
            Console.Read();
        }

        public static void Problem1()
        {
            /*
            double[,] transition =
            {  
                { 0.5, 0.4, 0.1 },
                { 0.1, 0.8, 0.1 },
                { 0.1, 0.6, 0.3 }
            };

            // Create the emission matrix B

            // High-Low
            double[,] emission = 
            {  
                { 0.1, 0.9 },
                { 0.5, 0.5 },
                { 0.9, 0.1 }
            };

            // Create the initial probabilities pi
            double[] initial =
            {
                0.3, 0.4, 0.3
            };

            // 0 mean High, 1 mean Low in sequence array
            int[] sequence = new int[] { 0, 0, 0 };

            HiddenMarkovModel hmm = new HiddenMarkovModel(transition, emission, initial);


            // Get log nature of probability
            double logLikeLihood = hmm.Evaluate(sequence);

            Console.Write("logLikeliHood = ln(prob) = " + logLikeLihood);
            Console.Read();
             */ 
            
            // Create the transition matrix A
            double[,] transition =
            {  
                { 0.7, 0.3 },
                { 0.4, 0.6 }                
            };

            // Create the emission matrix B
            // High-Low
            double[,] emission = 
            {  
                { 0.1, 0.4, 0.5 },
                { 0.7, 0.2, 0.1 }                
            };

            // Create the initial probabilities pi
            double[] initial =
            {
                0.6, 0.4
            };

            // Create a new hidden Markov model
            HiddenMarkovModel hmm = new HiddenMarkovModel(transition, emission, initial);

            // After that, one could, for example, query the probability
            // of a sequence occurring. We will consider the sequence
            int[] sequence = new int[] { 0, 1, 0, 2};

            double loglikelihood = hmm.Evaluate(sequence);
            Console.Write("Loglikelihood = ln(probability) = " + loglikelihood);
            //calculateForward(transition, emission, initial, sequence);

            Console.Read();
             
        }

        public static void calculateForward(double[,] transition, double[,] emission, double[] initial, int[] sequence)
        {   
            // Step 1 The first observation is High, so the initial probabilities are calculated by using
            // formula 2.17

            // 1.  Initialization
            int states = transition.GetLength(0);
            int T = sequence.Length;
            double[,] fwd = new double[states, T];
            for (int i = 0; i < states; i++)
                fwd[i, 0] = Math.Round(emission[i, 0] * initial[i] , 6);
            
            // 2. Induction

            for (int t = 1; t < T; t++)
            {                
                for (int i = 0; i < states; i++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < states; j++)
                        sum += fwd[j, t - 1] * transition[j, i] * emission[i,sequence[t]];
                    fwd[i, t] = Math.Round( sum, 6);
                }
            }

            double prob = 0.0;
            for (int i = 0; i < fwd.GetLength(0); i++)
            {
                prob += fwd[i, fwd.GetLength(0)-1];
            }
            Console.WriteLine("Prob = " + prob);
            Console.Read();

        }

        public static void Problem2()
        {
            // Create the transition matrix A
            double[,] transition =
            {  
                { 0.5, 0.4, 0.1 },
                { 0.1, 0.8, 0.1 },
                { 0.1, 0.6, 0.3 }
            };

            // Create the emission matrix B
            // High-Low
            double[,] emission = 
            {  
                { 0.1, 0.9 },
                { 0.5, 0.5 },
                { 0.9, 0.1 }
            };

            // Create the initial probabilities pi
            double[] initial =
            {
                0.3, 0.4, 0.3
            };
            
            // Create a new hidden Markov model
            HiddenMarkovModel hmm = new HiddenMarkovModel(transition, emission, initial);
                        
            // After that, one could, for example, query the probability
            // of a sequence occurring. We will consider the sequence
            int[] sequence = new int[] { 0, 0, 0 };
            double logLikelihood01;
            logLikelihood01 = hmm.Evaluate(sequence);
            Console.WriteLine(" Loglikelihood = ln(prob) = " + logLikelihood01);
            double logLikelihood;                       
            // We can also get the Viterbi path of the sequence
            int[] path = hmm.Decode(sequence, out logLikelihood);
            Console.WriteLine(" Loglikelihood = ln(prob) = " + logLikelihood);
            // 2 1 1 mean Sunny Cloudy Cloudy
            // 0 : Rainy
            // 1 : Cloudy
            // 2 : Sunny
            for (int i = 0; i < path.Length; i++)
            {
                Console.Write(path[i] + "   ");
            }
            Console.Read();
            
        }

        static void Main(string[] args)
        {
            //computeProbRandomIntSeq();
            //decodingSequence();
            //learningDiscreteByBaumWelch();
            //forwardProb();
            //learningClassifier();
            Problem1();
            //Problem2();
            //calculateForward(transition, emission, initial, sequence);            
        }
    }
}
