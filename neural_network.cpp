/*
Program   : neural_network.cpp
System    : ai
Function  : Demonstration of a single artificial neuron.
            A neuron is a fundamental component of artificial intelligence.
            
            Calculates the weighted sum, sigmoid activation,
            squared error loss and sigmoid derivative.
            
            Gradient descent and training will be added later.
            
            Test Data
            
            input 1   :  1.0
            input 2   :  0.0
            weight 1  :  0.8
            weight 2  : -0.4
            bias      : -0.1
            
            Expected Result

            Weighted Sum   = 0.7
            Sigmoid Output = 0.668188
            
            
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
#include <sstream>
#include <string>
#include <vector>

// Structure holding the input data, weights  and bias
struct neuron_data {
  double input_1;
  double input_2;
  double weight_1;
  double weight_2;
  double bias;
};

double calculate_weighted_sum(const neuron_data& nd) {
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

  return (nd.input_1 * nd.weight_1) + (nd.input_2 * nd.weight_2) + nd.bias;
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
  
  return 1/(1+std::exp(weighted_sum*-1));

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

void get_values(neuron_data& nd){
  nd.input_1   = prompt_double("Enter Input 1");
  nd.input_2   = prompt_double("Enter  Input 2");
  nd.weight_1  = prompt_double("Enter Weight 1");
  nd.weight_2  = prompt_double("Enter Weight 2");
  nd.bias      = prompt_double("Enter Bias");
}

int main(int argc, char* argv[]) {
  // easy to use: args as vector<string>
  std::vector<std::string> args(argv, argv + argc);
  neuron_data nd;

  // simple help
  if (argc > 1 && (args[1] == "-h" || args[1] == "--help")) {
    std::cout << "Usage: " << args[0] << " [options]" << std::endl
              << "  [-h | --help]" << std::endl;
    return EXIT_SUCCESS;
  }

  get_values(nd);
  double weighted_sum=calculate_weighted_sum(nd);       // Calculated the weighted sum of the neuron's inputs
  double sigmoid_output = sigmoid(weighted_sum);         // Apply the sigmoid activation function to the weighted sum to return number between 0 and 1
  double target = 1.0;
  double loss = calculate_loss(sigmoid_output, target);  // Calculate the loss as squared error (difference) between prediction and target
  double sigmoid_slope = sigmoid_derivative(sigmoid_output); // Calculate derivative of sigmoid function, used for gradient descent calc
  
  std::cout << "Weighted Sum = " << weighted_sum << std::endl;
  std::cout << "Sigmoid Output = " << sigmoid_output << std::endl;
  std::cout << "Loss = " << loss << std::endl;
  std::cout << "Sigmoid Slope = " << sigmoid_slope << std::endl;
  return EXIT_SUCCESS;
}

