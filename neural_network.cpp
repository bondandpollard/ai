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
              Weighted Sum    = 0.7
              Sigmoid Output  = 0.668188
              Loss            = 0.110099
              Sigmoid Slope   = 0.221713
              
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

// Structure holding the gradients used by gradient descent.
// Each value represents how the loss changes with respect
// to the corresponding trainable parameter.
struct neuron_gradients {
  double weight_1;
  double weight_2;
  double bias;
};

// Structure holding the calculated neuron data.
struct neuron_results {
  double weighted_sum;
  double sigmoid_output;
  double loss;
  double sigmoid_slope;
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
  // s = 1 / (1 + e^(-z))
  //
  // s = sigmoid output
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

double sigmoid_derivative( double sigmoid_output) {
  // Calculate the derivative of the sigmoid function.
  //
  // ds/dz = s * (1 - s)
  //
  // s     = sigmoid output
  // z     = weighted sum
  // ds/dz = rate of change of s with respect to z
  //
  // Used during gradient descent to calculate how changes
  // to the weighted sum affect the neuron's output.
  return sigmoid_output*(1-sigmoid_output);
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
  std::cout << "Sigmoid Output = " << nr.sigmoid_output << std::endl;
  std::cout << "Loss = " << nr.loss << std::endl;
  std::cout << "Sigmoid Slope = " << nr.sigmoid_slope << std::endl;
}

int main(int argc, char* argv[]) {
  // easy to use: args as vector<string>
  std::vector<std::string> args(argv, argv + argc);
  training_example te{};
  neuron_parameters np{};
  neuron_results nr{};

  // simple help
  if (argc > 1 && (args[1] == "-h" || args[1] == "--help")) {
    std::cout << "Usage: " << args[0] << " [options]" << std::endl
              << "  [-h | --help]" << std::endl;
    return EXIT_SUCCESS;
  }

  set_training_example(te);                                         // Set initial values of training data.
  set_neuron_parameters(np);                                        // Set initial neuron parameters: weights and bias.
  nr.weighted_sum   = calculate_weighted_sum(te, np);               // Calculate the weighted sum of the neuron's inputs.
  nr.sigmoid_output = sigmoid(nr.weighted_sum);                     // Apply the sigmoid activation function to the weighted sum to return a number between 0 and 1. 
  nr.loss           = calculate_loss(nr.sigmoid_output, te.target); // Calculate the loss as squared error (difference) between prediction and target.
  nr.sigmoid_slope  = sigmoid_derivative(nr.sigmoid_output);        // Calculate derivative of sigmoid function, used for gradient descent calculation.
  
  display_neuron(nr);
  
  return EXIT_SUCCESS;
}

