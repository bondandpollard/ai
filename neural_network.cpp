/*
Program   : neural_network.cpp
System    : ai
Function  : Demonstration of a neural network that can be trained.
            This is the foundation of Artificial Intelligence.
            
            Test Data
            
            input 1   :  1.0
            input 2   :  0.0
            weight 1  :  0.8
            weight 2  : -0.4
            bias      : -0.1
            
            Expected Result

            Weighted Sum  = 0.7
            Sigma         = 0.668188
            
            
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

double prompt_double (const std::string& p_prompt) {
  // Prompt user to enter a double (floating point) number.
  double value;
  std::cout << p_prompt << ":";
  while (!(std::cin >> value)) {   // handle bad input   
    std::cout << "Error: You must enter a valid floating point number: "; 
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  } 
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // remove newline from buffer so next prompt works
  return value;
}
 
double calculate_weighted_sum(const neuron_data nd) {
  // Calculate and return the weighted sum. 
  
  // z = x1*w1 + x2*w2 + b
  
  // x1 = input 1
  // x2 = input 2
  // w1 = weight 1
  // w2 = weight 2
  // b  = bias
  
  double result;

  return result = (nd.input_1 * nd.weight_1) + (nd.input_2 * nd.weight_2) + nd.bias;
}


double sigmoid(double weighted_sum) {
  // Calculate and return the sigmoid of weighted_sum.
  
  // s = 1 / (1+e^-z)
  
  // s = sigma
  // e = exponent
  // z = weighted sum (inverse)
  
  return 1/(1+std::exp(weighted_sum*-1));

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
  double weighted_sum=calculate_weighted_sum(nd);
  double neuron_output = sigmoid(weighted_sum);
  std::cout << "Weighted Sum = " << weighted_sum << '\n';
  std::cout << "Neuron Output = " << neuron_output << '\n';
  return EXIT_SUCCESS;
}

