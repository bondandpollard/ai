/*
Program   : nn_dataset.cpp
System    : ai
Function  : Neural Network Dataset Training.
            Demonstration of training a single artificial neuron using
            multiple training examples.

            The program builds on nn_neuron.cpp by introducing a dataset
            containing multiple input and target values. The neuron learns
            from each training example by using forward propagation,
            backpropagation and gradient descent.

            The program demonstrates:

              1. Dataset training
                 - store multiple training examples
                 - process each training example
                 - repeat training over multiple epochs

              2. Forward propagation
                 - calculate weighted sum
                 - apply sigmoid activation
                 - calculate prediction loss

              3. Backpropagation
                 - calculate loss slope
                 - calculate prediction slope
                 - apply the chain rule
                 - calculate parameter gradients

              4. Gradient descent
                 - update the weights and bias using the gradients
                   and learning rate

              5. Training feedback
                 - monitor loss during training
                 - observe how the neuron learns from multiple examples
              
Author    : Ian Bond
Date      : 28 September 2026

MODIFICATION HISTORY
Version   Log/Ref   Date        Author        Description
==========================================================================================================
1.00      IAB001    28/09/2026  Ian Bond      Created
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

void set_training_dataset(std::vector<training_example>& training_data) {

  // Create the training dataset for the logical OR function.
  //
  // x1  x2  target
  //  0   0     0
  //  0   1     1
  //  1   0     1
  //  1   1     1

  training_data = {
    {0.0, 0.0, 0.0},
    {0.0, 1.0, 1.0},
    {1.0, 0.0, 1.0},
    {1.0, 1.0, 1.0}
  };
}

void set_neuron_parameters(neuron_parameters& np){
  np.weight_1  = prompt_double("Enter Weight 1");
  np.weight_2  = prompt_double("Enter Weight 2");
  np.bias      = prompt_double("Enter Bias");
}

void display_training_example(const training_example& te) {
  
  std::cout << "Input 1 = "
            << te.input_1 << std::endl;

  std::cout << "Input 2 = "
            << te.input_2 << std::endl;

  std::cout << "Target = "
            << te.target << std::endl; 
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
  std::vector<training_example> 
    training_data;                 // Vector of training_example structures containing the input values and target outputs.
  neuron_parameters np{};          // Weights and bias (trainable parameters)
  neuron_results nr{};             // Forward pass and backpropagation calculation results
  neuron_gradients ng{};           // Gradients of loss with respect to each trainable parameter.
  double learning_rate = 0.1;      // Set the learning rate (eta), which controls the size of each gradient descent update to the weights and bias.
  double previous_loss;            // Loss before the current parameter update, used to measure improvement.
  int training_epochs = 10;        // Number of complete passes through the dataset.
  

  // Display command-line help if requested.
  if (argc > 1 && (args[1] == "-h" || args[1] == "--help")) {
    std::cout << "Usage: " << args[0] << " [options]" << std::endl
              << "  [-h | --help]" << std::endl;
    return EXIT_SUCCESS;
  }

  std::cout << "Neural Network - Dataset Training" << std::endl;
  
  // -----------------------------------------------------------------------------
  // INITIALIZE
  // -----------------------------------------------------------------------------
  
  set_training_dataset(training_data);                                   // Set values for the training dataset.
  set_neuron_parameters(np);                                             // Set initial neuron parameters: weights and bias.
  std::cout << std::endl << "INITIAL PARAMETER VALUES" << std::endl;
  display_parameters(np);                                                // Display the initial neuron parameters before training.    


  for (int epoch = 1; epoch <= training_epochs; ++epoch) {

    std::cout << std::endl
              << "TRAINING EPOCH " << epoch
              << std::endl;
              
    // Process each training example in the dataset.
    // One complete pass through the dataset constitutes one epoch.
    
    // te, nr and ng represent the current training example and its temporary
    // working data. np persists across all training examples and epochs.
    for (const training_example& te : training_data) {
      
      std::cout << std::endl << "TRAINING EXAMPLE" << std::endl;
      display_training_example(te);                                        // Show which example is being trained.
      
      // -----------------------------------------------------------------------------
      // FORWARD PASS
      //
      // Read:
      //   te - current training example
      //   np - current neuron parameters
      //
      // Update:
      //   nr - weighted sum, prediction and loss for the current training example
      //
      // No neuron parameters are changed during the forward pass.
      // -----------------------------------------------------------------------------
      std::cout << std::endl << "FORWARD PASS" << std::endl;
      forward_pass(te, np, nr);                                            // Perform the forward pass for the current training example using the current neuron parameters.
      display_forward_pass(nr);
      
      // -----------------------------------------------------------------------------
      // BACKPROPAGATION
      //
      // Read:
      //   te - current training example
      //   nr - prediction calculated by the forward pass
      //
      // Update:
      //   nr - loss slope, prediction slope and weighted sum slope
      //   ng - gradients for weight 1, weight 2 and bias
      //
      // No neuron parameters are changed during backpropagation.
      // -----------------------------------------------------------------------------
      std::cout << std::endl << "BACKPROPAGATION" << std::endl;
      backpropagate(te, nr, ng);                                           // Backpropagate the current prediction error and calculate parameter gradients.
      display_backpropagation(nr);                                         // Display the backpropagation calculation results.
      display_gradients(ng);                                               // Display the gradients for the trainable neuron parameters.
  
      // -----------------------------------------------------------------------------
      // GRADIENT DESCENT
      //
      // Read:
      //   nr - current loss, saved for comparison after the parameter update
      //   np - current weights and bias
      //   ng - gradients calculated by backpropagation
      //   learning_rate - controls the size of the parameter update
      //
      // Update:
      //   previous_loss - stores the loss before the parameter update
      //   np - weights and bias
      //
      // This is the stage where the neuron actually learns by changing its
      // trainable parameters.
      // Adjust the weights and bias in the direction that reduces the loss.
      // The gradient points towards increasing loss, so gradient descent
      // subtracts the gradient, scaled by the learning rate.
      // -----------------------------------------------------------------------------
      std::cout << std::endl << "GRADIENT DESCENT" << std::endl;
      previous_loss = nr.loss;
      update_parameters(np, ng, learning_rate);                            // Update the weights and bias using gradient descent and the calculated gradients.                       
      display_parameters(np);                                              // Display the updated neuron parameters after the gradient descent step.    

      // -----------------------------------------------------------------------------
      // FEEDBACK (FORWARD PASS AFTER UPDATE)
      //
      // Read:
      //   te - same current training example
      //   np - newly updated neuron parameters
      //
      // Update:
      //   nr - weighted sum, prediction and loss recalculated using
      //        the updated neuron parameters
      //
      // This allows the loss before and after the parameter update to be compared.
      // Run the forward pass again using the updated weights and bias.
      // With a suitable learning rate, the updated parameters should produce
      // a lower loss for the current training example.
      // -----------------------------------------------------------------------------
      std::cout << std::endl << "FORWARD PASS AFTER UPDATE" << std::endl;
      forward_pass(te, np, nr);                                            // Perform another forward pass using the updated neuron parameters.
      display_forward_pass(nr);                                            // Display results of the forward pass using the updated parameters.
      std::cout << std::endl << "LOSS COMPARISON:" << std::endl;           // Compare the loss before and after updating the parameters for the current training example.
      std::cout << "Previous Loss = " << previous_loss << std::endl;
      std::cout << "New Loss = " << nr.loss << std::endl; 
    }
  }
  
  return EXIT_SUCCESS;
}

