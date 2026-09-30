/*
Program   : nn_or.cpp
System    : ai
Function  : Neural Network Logical OR Dataset Training.
            Demonstration of training a single artificial neuron to learn the
            logical OR function from multiple training examples.

            The program builds on nn_neuron.cpp by introducing a dataset
            containing multiple input and target values. The neuron learns
            from each training example by using forward propagation,
            backpropagation and gradient descent.

            The program demonstrates:

              1. Dataset training
                 - store multiple training examples
                 - process each training example
                 - repeat training over multiple epochs

              2. Forward propagation (Forward pass)
                 - calculate weighted sum
                 - apply sigmoid activation to derive the prediction value
                 - calculate loss from prediction and target

              3. Backpropagation
                 - calculate loss slope
                 - calculate prediction slope
                 - calculate weighted sum slope
                 - apply the chain rule
                 - calculate parameter gradients

              4. Gradient descent
                 - update the weights and bias using the gradients
                   and learning rate

              5. Final evaluation
                - apply the trained parameters to every example
                - compare predictions with target values
                - evaluate the learned OR relationship without changing the parameters
                 
            This program demonstrates how a single artificial neuron can learn a relationship from multiple training examples.

            The neuron does not store a separate answer for each example. Instead, it has one set of trainable parameters: two weights and a bias. 
            The same parameters are applied to each different pair of input values to calculate a weighted sum, which is then converted by the sigmoid function 
            into a prediction between 0 and 1.
            
            training_example  : contains the input values and target output for the current training example.
            neuron_parameters : contains the weights and bias that represent what the neuron has learned.
            neuron_results    : contains the calculations for the current training example, including the weighted sum, prediction, loss and slopes.
            neuron_gradients  : contains the gradients calculated by backpropagation and used by gradient descent to update the neuron parameters.

            During training, the prediction is compared with the target value for the current example. 
            Backpropagation calculates how the loss changes with respect to each weight and the bias, and gradient descent adjusts those parameters.
            Repeating this process across the whole dataset and over many epochs causes one set of parameters to develop that works across all the training examples.
           
            In this program, the training_example dataset represents the logical OR function. 
            The neuron learns numerical parameter values that reproduce the OR relationship, rather than the OR rule being explicitly programmed.
            
            In an example run starting with weight 1 = 0.8, weight 2 = -0.4 and bias = -0.1, after 1000 epochs (complete passes through
            the dataset) the results are as follows:
            
            NB: The data represents logical OR: 0 OR 0 = 0, 0 OR 1 = 1, 1 OR 0 = 1, 1 OR 1 = 1.
            
            Input 1     Input 2   Target    Prediction
            0           0         0         0.136448
            0           1         1         0.916753
            1           0         1         0.917035
            1           1         1         0.998704
            
              
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
#include <stdexcept>
#include <string>
#include <vector>

bool verbose = false;     // Switch detailed output on/off

// Structure holding one training example:
//
// Input values and the expected target output.
//
// Each instance contains the input values presented to the neuron and the
// expected target output for those inputs.
//
// During training, one training_example is processed at a time.
// Function forward_pass() uses input_1 and input_2 with the current neuron parameters
// to calculate a prediction (stored in neuron_results). 
//
// The prediction is then compared with target to calculate the loss used during backpropagation.
struct training_example {
  double input_1;
  double input_2;
  double target;
};

// Structure holding the weights and bias (trainable parameters).
//
// The neuron_parameters instance persists throughout training.
// Its weights and bias represent what the neuron has learned so far.
//
// For each training example, forward_pass() combines the current input values
// with these parameters to calculate neuron_results, including the weighted
// sum, prediction (sigmoid output) and loss.
//
// NB: nr.prediction is the neuron's actual output, which is compared
// with the target output during training.
//
// Backpropagation and gradient descent then adjust these parameters so that
// future predictions should move closer to the target values.
struct neuron_parameters {
  double weight_1;
  double weight_2;
  double bias;
};

// Structure holding the gradients calculated by backpropagation.
//
// Each gradient represents how the loss changes with respect to one
// trainable neuron parameter:
//
//   weight_1 - dL/dw1
//   weight_2 - dL/dw2
//   bias     - dL/db
//
// The sign and size of each gradient indicate the direction and rate at
// which the loss changes as the corresponding parameter changes.
//
// Gradient descent uses these values, together with the learning rate,
// to adjust the neuron parameters in the direction that reduces the loss.
struct neuron_gradients {
  double weight_1;    // dL/dw1
  double weight_2;    // dL/dw2
  double bias;        // dL/db
};

// Structure holding calculated neuron results for the current training example.
//
// The forward pass stores:
//   weighted_sum - z, calculated from the current inputs, weights and bias
//   prediction   - p, sigmoid output between 0 and 1
//   loss         - L, squared error between prediction and target
//
// Backpropagation then stores the corresponding slopes:
//   loss_slope         - dL/dp, how loss changes with prediction
//   prediction_slope   - dp/dz, how prediction changes with weighted sum
//   weighted_sum_slope - dL/dz, how loss changes with weighted sum
//
// These values are temporary working results for the current example.
// They are recalculated as each training example is processed.
struct neuron_results {
  double weighted_sum;
  double weighted_sum_slope;
  double prediction;      // Sigmoid output between 0 and 1, derived from the current input values and neuron parameters.
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
  
  // Calculate loss as the squared difference between prediction and target.
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
                                                                         
  nr.loss = calc_loss(nr.prediction, te.target);           // Calculate the loss as squared difference between prediction and target.
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
  std::cout << p_prompt << " : ";
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

  // Prompt the user to enter the initial values for the neuron's
  // trainable parameters: two weights and a bias.
  //
  // These values provide the starting point for training.
  // During gradient descent, the parameters are repeatedly adjusted
  // using the calculated gradients so that the neuron's predictions
  // move closer to the target values.
  
  np.weight_1  = prompt_double("Enter Weight 1");
  np.weight_2  = prompt_double("Enter Weight 2");
  np.bias      = prompt_double("Enter Bias");
}

void display_training_example(const training_example& te) {
  
  // Display the current training example.
  //
  // input_1 and input_2 are the values presented to the neuron.
  // target is the expected output that the neuron's prediction
  // will be compared with during training.
  
  if (verbose) {
    std::cout << std::endl << "TRAINING EXAMPLE" << std::endl;
    std::cout << "Input 1 = " << te.input_1 << std::endl;
    std::cout << "Input 2 = " << te.input_2 << std::endl;
    std::cout << "Target = " << te.target << std::endl; 
  }
}

void display_forward_pass(const neuron_results& nr) {
  // Display the calculated results of the forward pass for the current
  // training example.
  //
  // weighted_sum - result of combining the input values with the current
  //                weights and bias
  // prediction   - sigmoid output between 0 and 1 produced from weighted_sum
  // loss         - squared error showing how far the prediction is from
  //                the target value

  if (verbose) {
    std::cout << std::endl << "FORWARD PASS" << std::endl;
    std::cout << "Weighted Sum = " << nr.weighted_sum << std::endl;
    std::cout << "Prediction (Sigmoid Output) = " << nr.prediction << std::endl;
    std::cout << "Loss = " << nr.loss << std::endl;
  } 
}

void display_backpropagation(const neuron_results& nr) {
  
  // Display the derivative values calculated during backpropagation.
  //
  // loss_slope         - dL/dp: how the loss changes as the prediction changes
  // prediction_slope   - dp/dz: how the prediction changes as the weighted sum changes
  // weighted_sum_slope - dL/dz: how the loss changes as the weighted sum changes
  //
  // The weighted_sum_slope is obtained by applying the chain rule:
  //   dL/dz = (dL/dp) * (dp/dz)
  //
  // These values are then used to calculate the gradients for the
  // neuron's trainable parameters.
  
  
  if (verbose) {
    std::cout << std::endl << "BACKPROPAGATION" << std::endl;
    std::cout << "Loss Slope = " << nr.loss_slope << std::endl;
    std::cout << "Prediction Slope = " << nr.prediction_slope << std::endl;
    std::cout << "Weighted Sum Slope = " << nr.weighted_sum_slope << std::endl;
  }
}

void display_gradients(const neuron_gradients& ng) {
  
  // Display the parameter gradients calculated by backpropagation.
  //
  // Each gradient shows how the loss changes with respect to one
  // trainable neuron parameter:
  //
  //   weight_1 gradient - dL/dw1
  //   weight_2 gradient - dL/dw2
  //   bias gradient     - dL/db
  //
  // The sign and size of each gradient determine how gradient descent
  // will adjust the corresponding parameter to reduce the loss.
  
  if (verbose) {
    std::cout << "GRADIENTS:" << std::endl;
    std::cout << "Weight 1 Gradient = " << ng.weight_1 << std::endl;
    std::cout << "Weight 2 Gradient = " << ng.weight_2 << std::endl;
    std::cout << "Bias Gradient = " << ng.bias << std::endl;
  }
}

void display_parameters(const neuron_parameters& np) {

  // Display the neuron's current trainable parameter values.
  //
  // weight_1 and weight_2 control the influence of the two input values.
  // bias shifts the weighted sum independently of the inputs.
  //
  // These parameters persist throughout training and are adjusted by
  // gradient descent. Their current values represent what the neuron
  // has learned so far.

  std::cout << std::endl << "PARAMETER VALUES" << std::endl;
  std::cout << "Weight 1 = " << np.weight_1 << std::endl;
  std::cout << "Weight 2 = " << np.weight_2 << std::endl;
  std::cout << "Bias = " << np.bias << std::endl;
}

void evaluate_dataset(const std::vector<training_example>& training_data,
                      const neuron_parameters& np) {

  // Evaluate the trained neuron using every example in the dataset.
  //
  // The final trained weights and bias are displayed once, then the same
  // neuron parameters are applied to each training example using a forward pass.
  //
  // For each example, the neuron calculates a new weighted sum, prediction
  // and loss from the example inputs and the trained parameters.
  //
  // No backpropagation or gradient descent is performed during evaluation,
  // so the neuron parameters are not changed.
  //
  // This shows how well the final trained neuron reproduces the logical OR
  // relationship represented by the dataset.

  neuron_results nr{};

  std::cout << std::endl << "FINAL EVALUATION" << std::endl;
  
  display_parameters(np);
  
  std::cout << std::endl << "PREDICTIONS:" << std::endl;
  
  for (const training_example& te : training_data) {
    forward_pass(te, np, nr);
    std::cout << "[" << te.input_1 << ", " << te.input_2 << "] target " << te.target << " -> " << nr.prediction << std::endl;  
  }
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
  int training_epochs = 1000;      // Number of complete passes through the dataset.
  
  // Check command line arguments
  if (argc > 1) {
  
    // Display command-line help if requested.
    if (args[1] == "-h" || args[1] == "--help") {
      std::cout << "Usage: " << args[0] << " [number of epochs] [-v | --verbose]" << std::endl
                << "  [-h | --help]" << std::endl;
      return EXIT_SUCCESS;
    }
    
    // Set training_epochs from command line argument if a number is provided
    try {
      std::size_t pos = 0;

      training_epochs = std::stoi(args[1], &pos);

      // Ensure the whole argument was a valid integer.
      if (pos != args[1].size()) {
        throw std::invalid_argument("invalid integer");
      }

      if (training_epochs <= 0) {
        throw std::invalid_argument("epochs must be greater than zero");
      }
    }
    catch (const std::invalid_argument&) {
      std::cerr << "Error: epochs must be a positive integer." << std::endl;
      return EXIT_FAILURE;
    } 
    catch (const std::out_of_range&) {
      std::cerr << "Error: epochs value is too large." << std::endl;
      return EXIT_FAILURE;
    }
    
    // Check if verbose output has been specified as an argument
    if (argc > 2) {
      if (args[2] == "-v" || args[2] == "--verbose") {
        verbose = true;
      }
    }
  }

  std::cout << "Neural Network Logical OR Dataset Training" << std::endl;
  
  // -----------------------------------------------------------------------------
  // INITIALIZE
  // -----------------------------------------------------------------------------
  
  set_training_dataset(training_data);                                   // Set values for the training dataset.
  set_neuron_parameters(np);                                             // Set initial neuron parameters: weights and bias.
  std::cout << std::endl;
  if (verbose) {display_parameters(np);}                                 // Display the initial neuron parameters before training.    

  for (int epoch = 1; epoch <= training_epochs; ++epoch) {
    
    double epoch_loss = 0.0;                                               // Sum of the losses for all training examples in the current epoch.

    if (verbose) {
      std::cout << std::endl
                << "TRAINING EPOCH " << epoch << " of " << training_epochs
                << std::endl;
    }
              
    // Process each training example in the dataset.
    // One complete pass through the dataset constitutes one epoch.
    
    // te, nr and ng represent the current training example and its temporary
    // working data. np persists across all training examples and epochs.
    for (const training_example& te : training_data) {
      
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
      forward_pass(te, np, nr);                                            // Perform the forward pass for the current training example using the current neuron parameters.
      display_forward_pass(nr);
      
      epoch_loss += nr.loss;                                               // Add the current training example's loss to the epoch total.
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
      backpropagate(te, nr, ng);                                           // Backpropagate the current prediction error and calculate parameter gradients.
      display_backpropagation(nr);                                         // Display the backpropagation calculation results.
      display_gradients(ng);                                               // Display the gradients for the trainable neuron parameters.
  
      // -----------------------------------------------------------------------------
      // GRADIENT DESCENT
      //
      // Read:
      //   np - current weights and bias
      //   ng - gradients calculated by backpropagation
      //   learning_rate - controls the size of the parameter update
      //
      // Update:
      //   np - weights and bias
      //
      // This is the stage where the neuron actually learns by changing its
      // trainable parameters.
      // Adjust the weights and bias in the direction that reduces the loss.
      // The gradient points towards increasing loss, so gradient descent
      // subtracts the gradient, scaled by the learning rate.
      // -----------------------------------------------------------------------------
      if (verbose) {std::cout << std::endl << "GRADIENT DESCENT" << std::endl;}
      update_parameters(np, ng, learning_rate);                            // Update the weights and bias using gradient descent and the calculated gradients.                       
      if (verbose) {display_parameters(np);}                               // Display the updated neuron parameters after the gradient descent step.    

    }
    
    double average_loss = epoch_loss / training_data.size();               // Mean loss across all training examples in the current epoch.

    if (verbose) {
      std::cout << std::endl 
                << "EPOCH "
                << epoch << " AVERAGE LOSS = "
                << average_loss
                << std::endl;
    }
    else {
      std::cout << "TRAINING EPOCH "
                << epoch << " of " << training_epochs
                << " Average Loss = "
                << average_loss
                << std::endl;
    }
  }
  
  // -----------------------------------------------------------------------------
  // FINAL EVALUATION
  //
  // Evaluate the trained neuron using every example in the dataset.
  // The trained neuron parameters are not changed during evaluation.
  // Only a forward pass is required.
  // -----------------------------------------------------------------------------

  evaluate_dataset(training_data, np);
  
  return EXIT_SUCCESS;
}