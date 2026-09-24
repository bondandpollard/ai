/*
Program   : neural_network.cpp
System    : ai
Function  : Demonstration of a single artificial neuron.
            An artificial neuron is a fundamental component of neural networks.

            Test Training Example
              input 1 : 1.0
              input 2 : 0.0
              target  : 1.0

            Initial Neuron Parameters
              weight 1 :  0.8
              weight 2 : -0.4
              bias     : -0.1

            Expected Result
              Weighted Sum                = 0.7
              Prediction (Sigmoid Output) = 0.668188
              Loss                        = 0.110099
              Prediction Slope            = 0.221713
              Loss Slope                  = -0.663624
              Weighted Sum Slope          = -0.147134
              Weight 1 Gradient           = -0.147134
              Weight 2 Gradient           = 0
              Bias Gradient               = -0.147134
              
Author    : Ian Bond
Date      : 18 September 2026

MODIFICATION HISTORY
Version   Log/Ref   Date        Author        Description
==========================================================================================================
1.00      IAB001    18/09/2026  Ian Bond      Created
*/

#include <cmath>
#include <cstdlib>   // EXIT_SUCCESS / EXIT_FAILURE
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// Structure holding one training example:
// input values and the expected target output.
struct training_example {
  double input_1;
  double input_2;
  double target;
};

// Structure holding the weights and bias (trainable parameters).
struct neuron_parameters {
  double weight_1;
  double weight_2;
  double bias;
};

// Structure holding the gradients calculated by backpropagation
// and used by gradient descent.
// Each value represents how the loss changes with respect
// to the corresponding trainable parameter.
struct neuron_gradients {
  double weight_1;    // dL/dw1
  double weight_2;    // dL/dw2
  double bias;        // dL/db
};

// Structure holding the calculated neuron data.
struct neuron_results {
  double weighted_sum;
  double weighted_sum_slope;
  double prediction;      // sigmoid output
  double prediction_slope;
  double loss;
  double loss_slope;

};

double calculate_weighted_sum(const training_example& te, const neuron_parameters& np) {
  // Calculate the weighted sum of the neuron's inputs.
  //
  // z = (x1 * w1) + (x2 * w2) + b
  //
  // z  = weighted sum
  // x1 = input 1
  // x2 = input 2
  // w1 = weight 1
  // w2 = weight 2
  // b  = bias

  return (te.input_1 * np.weight_1) + (te.input_2 * np.weight_2) + np.bias;
}


double sigmoid(double weighted_sum) {
  // Apply the sigmoid activation function.
  //
  // p = 1 / (1 + e^(-z))
  //
  // p = prediction (sigmoid output)
  // e = Euler's number (approximately 2.71828)
  // z = weighted sum
  //
  // Returns a value between 0 and 1.
  
  return 1 / (1 + std::exp(-weighted_sum));

}

double calculate_loss(double prediction, double target) {
  // Calculate the squared error between prediction and target.
  //
  // L = (t - p)^2
  //
  // L = loss
  // t = target (correct answer)
  // p = prediction (sigmoid output)
  //
  // A lower loss indicates a prediction closer to the target.
  return (target - prediction) * (target - prediction);
}

double sigmoid_derivative(double prediction) {
  // Calculate the derivative of the sigmoid function.
  //
  // dp/dz = p * (1 - p)
  //
  // p     = prediction (sigmoid output)
  // z     = weighted sum
  // dp/dz = rate of change of prediction with respect to weighted sum
  //  
  // Used during backpropagation to calculate how changes
  // to the weighted sum affect the neuron's prediction.
  return prediction * (1 - prediction);
}

double loss_derivative(double prediction, double target) {
  // Calculate the derivative of the squared error loss.
  //
  // dL/dp = 2 * (p - t)
  //
  // L     = loss
  // p     = prediction
  // t     = target
  // dL/dp = rate of change of loss with respect to prediction

  return 2 * (prediction - target);
}

double weighted_sum_derivative(double loss_slope, double prediction_slope) {
  // Calculate the derivative of the loss with respect to the weighted sum.
  //
  // dL/dz = (dL/dp) * (dp/dz)
  //
  // dL/dz = rate of change of loss with respect to weighted sum
  // dL/dp = loss slope
  // dp/dz = prediction slope
  //
  // Uses the chain rule to combine the two derivatives.

  return (loss_slope * prediction_slope);
}

void calculate_gradients(const training_example& te,
                         const neuron_results& nr,
                         neuron_gradients& ng) {

  // Calculate the gradients of the loss with respect to
  // each trainable neuron parameter.
  //
  // dL/dw1 = (dL/dz) * x1
  // dL/dw2 = (dL/dz) * x2
  // dL/db  = (dL/dz)
  //
  // dL/dz = weighted sum slope
  // x1     = input 1
  // x2     = input 2

  ng.weight_1 = nr.weighted_sum_slope * te.input_1;
  ng.weight_2 = nr.weighted_sum_slope * te.input_2;
  ng.bias     = nr.weighted_sum_slope;
}

double prompt_double (const std::string& p_prompt) {
  // Prompt user to enter a double (floating point) number.
  double value;
  std::cout << p_prompt << ":";
  while (!(std::cin >> value)) {   // handle bad input   
    std::cout << "Error: You must enter a valid floating point number: "; 
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  } 
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // discard remainder of input line
  return value;
}

void set_training_example(training_example& te){
  te.input_1   = prompt_double("Enter Input 1");
  te.input_2   = prompt_double("Enter  Input 2");
  te.target    = prompt_double("Enter Target");
}

void set_neuron_parameters(neuron_parameters& np){
  np.weight_1  = prompt_double("Enter Weight 1");
  np.weight_2  = prompt_double("Enter Weight 2");
  np.bias      = prompt_double("Enter Bias");
}

void display_neuron(const neuron_results& nr) {
  std::cout << "Weighted Sum = " << nr.weighted_sum << std::endl;
  std::cout << "Prediction (Sigmoid Output) = " << nr.prediction << std::endl;
  std::cout << "Loss = " << nr.loss << std::endl;
  std::cout << "Prediction Slope = " << nr.prediction_slope << std::endl;
  std::cout << "Loss Slope = " << nr.loss_slope << std::endl;
  std::cout << "Weighted Sum Slope = " << nr.weighted_sum_slope << std::endl;
}

void display_gradients(const neuron_gradients& ng) {
  // Display how the loss changes with respect to each
  // trainable neuron parameter.
  
  std::cout << "GRADIENTS:" << std::endl;
  std::cout << "Weight 1 = " << ng.weight_1 << std::endl;
  std::cout << "Weight 2 = " << ng.weight_2 << std::endl;
  std::cout << "Bias = " << ng.bias << std::endl;
}

int main(int argc, char* argv[]) {
  // easy to use: args as vector<string>
  std::vector<std::string> args(argv, argv + argc);
  training_example te{};
  neuron_parameters np{};
  neuron_results nr{};
  neuron_gradients ng{};
  

  // simple help
  if (argc > 1 && (args[1] == "-h" || args[1] == "--help")) {
    std::cout << "Usage: " << args[0] << " [options]" << std::endl
              << "  [-h | --help]" << std::endl;
    return EXIT_SUCCESS;
  }

  set_training_example(te);                                              // Set values for the training example.
  set_neuron_parameters(np);                                             // Set initial neuron parameters: weights and bias.
  nr.weighted_sum       = calculate_weighted_sum(te, np);                // Calculate the weighted sum of the neuron's inputs.
  nr.prediction         = sigmoid(nr.weighted_sum);                      // Apply the sigmoid activation function to the weighted sum to return a number between 0 and 1. 
  nr.loss               = calculate_loss(nr.prediction, te.target);      // Calculate the loss as squared error (difference) between prediction and target.
  nr.prediction_slope   = sigmoid_derivative(nr.prediction);             // Calculate the prediction slope (dp/dz) using the sigmoid derivative.
  nr.loss_slope         = loss_derivative(nr.prediction, te.target);     // Calculate how the loss changes with respect to the prediction.
  nr.weighted_sum_slope = weighted_sum_derivative(nr.loss_slope, 
                                                  nr.prediction_slope);  // Use the chain rule to calculate how the loss changes with respect to the weighted sum.
  calculate_gradients(te, nr, ng);                                       // Calculate the gradients of the loss with respect to weight 1, weight 2 and bias.
  display_neuron(nr);                                                    // Display the calculated neuron results.
  display_gradients(ng);                                                 // Display the gradients for the trainable neuron parameters.
  
  return EXIT_SUCCESS;
}

