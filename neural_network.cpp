/*
Program   : neural_network.cpp
System    : ai
Function  : Demonstration of a neural network that can be trained.
            This is the foundation of Artificial Intelligence.
Author    : Ian Bond
Date      : 18 September 2026

MODIFICATION HISTORY
Version   Log/Ref   Date        Author        Description
==========================================================================================================
1.00      IAB001    18/09/2026  Ian Bond      Created
*/

#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>   // EXIT_SUCCESS / EXIT_FAILURE

// Structure holding the input data, weights  and bias
struct neuron_data {
  double input_1;
  double input_2;
  double weight_1;
  double weight_2;
  double bias;
};


double prompt_double (const std::string& p_prompt) {
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
 
double calculate_neuron(const neuron_data nd) {    
  double result;
  // Calculate and return the weighted sum.
  return result = (nd.input_1 * nd.weight_1) + (nd.input_2 * nd.weight_2) + nd.bias;
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
  double neuron_output;

  // simple help
  if (argc > 1 && (args[1] == "-h" || args[1] == "--help")) {
    std::cout << "Usage: " << args[0] << " [options]" << std::endl
              << "  [-h | --help]" << std::endl;
    return EXIT_SUCCESS;
  }

  get_values(nd);
  neuron_output=calculate_neuron(nd);
  std::cout << "Neuron Output=" << neuron_output;
  
  return EXIT_SUCCESS;
}

