/*
Program   : neural_network.cpp
System    : ai
Function  : Demonstration of training a single artificial neuron using
            backpropagation and gradient descent.
            An artificial neuron is a fundamental component of neural networks.
            
            The program demonstrates:

              1. Forward propagation
                 - calculate weighted sum
                 - apply sigmoid activation
                 - calculate prediction loss

              2. Backpropagation
                 - calculate loss slope
                 - calculate prediction slope
                 - apply the chain rule
                 - calculate parameter gradients

              3. Gradient descent
                 - update the weights and bias using the gradients
                   and learning rate

            Test Training Example
              input 1 : 1.0
              input 2 : 0.0
              target  : 1.0

            Initial Neuron Parameters
              weight 1 :  0.8
              weight 2 : -0.4
              bias     : -0.1

            Expected Initial Result
            
            Forward Pass:
              Weighted Sum                =  0.7
              Prediction (Sigmoid Output) =  0.668188
              Loss                        =  0.110099
            
            Backpropagation:
              Loss Slope                  = -0.663624
              Prediction Slope            =  0.221713
              Weighted Sum Slope          = -0.147134
            
            Gradients:
              Weight 1 Gradient           = -0.147134
              Weight 2 Gradient           =  0
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

double calc_weighted_sum(const training_example& te, const neuron_parameters& np) {
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

double calc_prediction(double weighted_sum) {
  // Calculate prediction by applying the sigmoid activation function.
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

double calc_loss(double prediction, double target) {
  // Calculate loss as the squared error between prediction and target.
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

void forward_pass(const training_example& te, 
                  const neuron_parameters& np, 
                  neuron_results& nr) {
                    
  // Perform the forward pass using the current neuron parameters.
  // Calculate the weighted sum, prediction and resulting loss.

  nr.weighted_sum = calc_weighted_sum(te, np);             // Calculate the weighted sum of the neuron's inputs.
  
  nr.prediction = calc_prediction(nr.weighted_sum);        // Apply the sigmoid activation function to the weighted sum to return the 
                                                           // prediction as a number between 0 and 1. 
                                                                         
  nr.loss = calc_loss(nr.prediction, te.target);           // Calculate the loss as squared error (difference) between prediction and target.
}

double calc_loss_slope(double prediction, double target) {
  // Calculate loss slope as the derivative of the squared error loss.
  //
  // dL/dp = 2 * (p - t)
  //
  // L     = loss
  // p     = prediction
  // t     = target
  // dL/dp = rate of change of loss with respect to prediction

  return 2 * (prediction - target);
}

double calc_prediction_slope(double prediction) {
  // Calculate prediction slope as the derivative of the sigmoid function.
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

double calc_weighted_sum_slope(double loss_slope, double prediction_slope) {
  // Calculate weighted sum slope as the derivative of the loss with respect to the weighted sum.
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

void calc_gradients(const training_example& te,
                    const neuron_results& nr,
                    neuron_gradients& ng) {

  // Calculate the gradients of the loss with respect to
  // each trainable neuron parameter.
  //
  // dL/dw1 = (dL/dz) * x1
  // dL/dw2 = (dL/dz) * x2
  // dL/db  = (dL/dz)
  //
  // dL/dz  = weighted sum slope
  // x1     = input 1
  // x2     = input 2
  //
  // Each weight gradient is multiplied by its corresponding input
  // because that input determines how strongly the weight affects z.
  //
  // The bias gradient equals dL/dz because the bias is added directly
  // to z: dz/db = 1.
  
  ng.weight_1 = nr.weighted_sum_slope * te.input_1;
  ng.weight_2 = nr.weighted_sum_slope * te.input_2;
  ng.bias     = nr.weighted_sum_slope;
}

void backpropagate(const training_example& te,
                   neuron_results& nr,
                   neuron_gradients& ng) {
                     
  // Perform backpropagation for the current prediction error.
  // Calculate the slopes and parameter gradients needed for gradient descent.

  nr.loss_slope = calc_loss_slope(nr.prediction, te.target);  // Calculate how the loss changes with respect to the prediction.

  nr.prediction_slope = calc_prediction_slope(nr.prediction); // Calculate the prediction slope (dp/dz) using the sigmoid derivative.

  nr.weighted_sum_slope = calc_weighted_sum_slope(nr.loss_slope, nr.prediction_slope); // Use the chain rule to calculate how the loss changes 
                                                                                       // with respect to the weighted sum.

  calc_gradients(te, nr, ng); // Calculate the gradients of the loss with respect to weight 1, weight 2 and bias.
}

void update_parameters(neuron_parameters& np,
                       const neuron_gradients& ng,
                       double learning_rate) {

  // Update the trainable neuron parameters using gradient descent.
  //
  // new parameter = old parameter - (learning rate * gradient)
  //
  // w1 = w1 - (eta * dL/dw1)
  // w2 = w2 - (eta * dL/dw2)
  // b  = b  - (eta * dL/db)
  //
  // eta = learning rate.
  //
  // The gradient points in the direction of increasing loss.
  // Subtracting the gradient, scaled by the learning rate,
  // moves the parameter in the direction of decreasing loss.
  // The learning rate controls how large each update step is.


  np.weight_1 -= learning_rate * ng.weight_1;
  np.weight_2 -= learning_rate * ng.weight_2;
  np.bias     -= learning_rate * ng.bias;
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
  te.input_2   = prompt_double("Enter Input 2");
  te.target    = prompt_double("Enter Target");
}

void set_neuron_parameters(neuron_parameters& np){
  np.weight_1  = prompt_double("Enter Weight 1");
  np.weight_2  = prompt_double("Enter Weight 2");
  np.bias      = prompt_double("Enter Bias");
}

void display_forward_pass(const neuron_results& nr) {

  std::cout << "Weighted Sum = "
            << nr.weighted_sum << std::endl;

  std::cout << "Prediction (Sigmoid Output) = "
            << nr.prediction << std::endl;

  std::cout << "Loss = "
            << nr.loss << std::endl;
}

void display_backpropagation(const neuron_results& nr) {

  std::cout << "Loss Slope = "
            << nr.loss_slope << std::endl;

  std::cout << "Prediction Slope = "
            << nr.prediction_slope << std::endl;

  std::cout << "Weighted Sum Slope = "
            << nr.weighted_sum_slope << std::endl;
}

void display_gradients(const neuron_gradients& ng) {
  // Display how the loss changes with respect to each
  // trainable neuron parameter.
  
  std::cout << "GRADIENTS:" << std::endl;
  std::cout << "Weight 1 Gradient = " << ng.weight_1 << std::endl;
  std::cout << "Weight 2 Gradient = " << ng.weight_2 << std::endl;
  std::cout << "Bias Gradient = " << ng.bias << std::endl;
}

void display_parameters(const neuron_parameters& np) {

  std::cout << "PARAMETERS:" << std::endl;
  std::cout << "Weight 1 = " << np.weight_1 << std::endl;
  std::cout << "Weight 2 = " << np.weight_2 << std::endl;
  std::cout << "Bias = " << np.bias << std::endl;
}

int main(int argc, char* argv[]) {
  std::vector<std::string> 
    args(argv, argv + argc);       // Store command-line arguments as strings for easier processing
  training_example te{};           // Input values and target used for training.
  neuron_parameters np{};          // Weights and bias (trainable parameters)
  neuron_results nr{};             // Forward pass and backpropagation calculation results
  neuron_gradients ng{};           // Gradients of loss with respect to each trainable parameter.
  double learning_rate = 0.1;      // Set the learning rate (eta), which controls the size of each gradient descent update to the weights and bias.
  double previous_loss;            // Loss before the current parameter update, used to measure improvement.
  int training_iterations = 10;    // Number of training iterations to perform.
  

  // Display command-line help if requested.
  if (argc > 1 && (args[1] == "-h" || args[1] == "--help")) {
    std::cout << "Usage: " << args[0] << " [options]" << std::endl
              << "  [-h | --help]" << std::endl;
    return EXIT_SUCCESS;
  }

  // -----------------------------------------------------------------------------
  // INITIALIZE
  // -----------------------------------------------------------------------------
  
  set_training_example(te);                                              // Set values for the training example.
  set_neuron_parameters(np);                                             // Set initial neuron parameters: weights and bias.
  std::cout << std::endl << "INITIAL PARAMETER VALUES" << std::endl;
  display_parameters(np);                                                // Display the initial neuron parameters before training.    

  // -----------------------------------------------------------------------------
  // FORWARD PASS (INITIAL)
  // -----------------------------------------------------------------------------
  
  std::cout << std::endl << "FORWARD PASS" << std::endl;
  forward_pass(te, np, nr);                                              // Perform the initial forward pass using the starting neuron parameters.
  display_forward_pass(nr);                                              // Display results of initial forward pass.
  
  // Repeat backpropagation, gradient descent and feedback
  // for the required number of training iterations.
  for (int iteration = 1; iteration <= training_iterations; ++iteration) {

    std::cout << std::endl
              << "TRAINING ITERATION " << iteration
              << std::endl;
  
    // -----------------------------------------------------------------------------
    // BACKPROPAGATION
    // -----------------------------------------------------------------------------
  
    std::cout << std::endl << "BACKPROPAGATION" << std::endl;
    backpropagate(te, nr, ng);                                           // Backpropagate the current prediction error and calculate parameter gradients.
    display_backpropagation(nr);                                         // Display the backpropagation calculation results.
    display_gradients(ng);                                               // Display the gradients for the trainable neuron parameters.
  
    // -----------------------------------------------------------------------------
    // GRADIENT DESCENT
    //
    // Adjust the weights and bias in the direction that reduces the loss.
    // The gradient points towards increasing loss, so gradient descent
    // subtracts the gradient, scaled by the learning rate.
    // -----------------------------------------------------------------------------
  
    std::cout << std::endl << "GRADIENT DESCENT" << std::endl;
    previous_loss = nr.loss;
    update_parameters(np, ng, learning_rate);                            // Update the weights and bias using gradient descent and the calculated gradients.                       
    display_parameters(np);                                              // Display the updated neuron parameters after the gradient descent step.    

    // -----------------------------------------------------------------------------
    // FEEDBACK
    //
    // Run the forward pass again using the updated weights and bias.
    // If gradient descent has moved the parameters in the correct direction,
    // the new loss should be lower than the previous loss.
    // With a suitable learning rate, the updated parameters should produce
    // a lower loss than before the gradient descent step.
    // -----------------------------------------------------------------------------

    std::cout << std::endl << "FORWARD PASS" << std::endl;
    forward_pass(te, np, nr);                                            // Perform another forward pass using the updated neuron parameters.
    display_forward_pass(nr);                                            // Display results of the forward pass using the updated parameters.
    std::cout << std::endl << "LOSS COMPARISON:" << std::endl;           // Compare the loss before and after the current training iteration.
    std::cout << "Previous Loss = " << previous_loss << std::endl;
    std::cout << "New Loss = " << nr.loss << std::endl;
  }
  
  return EXIT_SUCCESS;
}

