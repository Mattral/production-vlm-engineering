
# Training Neural Networks

Training neural networks is the process of teaching a model to make accurate predictions by learning from data. This involves iterative steps of forward and backward propagation, optimization, and the use of loss functions to quantify the model's performance. In this note, we'll delve into the training process, introducing mathematical concepts gradually.

## Forward Propagation

Forward propagation is the initial phase of training in which data is passed through the network to compute predictions. It involves several key steps:

1. **Input Layer:** The raw input data, often represented as a feature vector, is provided to the input layer of the neural network.

2. **Weighted Sum:** In each neuron (or node), a weighted sum of the input data is computed using the following equation:
   
   Weighted Sum=∑ 
i=1 
​
 (w 
i
​
 ⋅x 
i
​
 )+b

   Where:
   - Weighted Sum is the result of the weighted sum.
   - \(w_i\) are the weights.
   - \(x_i\) are the input values.
   - \(b\) is the bias term.

3. **Activation Function:** After computing the weighted sum, an activation function is applied to introduce non-linearity. Common activation functions include the sigmoid, ReLU, and tanh functions.

4. **Output:** The result of the activation function is the output of the neuron and serves as the input to the subsequent layer.

This process continues through each layer until the final output is computed.

## Backward Propagation

Backward propagation, often referred to as backpropagation, is the phase of training where the model learns from its errors. It involves calculating gradients with respect to model parameters (weights and biases) and using them to adjust these parameters. Key steps include:

1. **Loss Function:** A loss function is used to measure the difference between the model's predictions and the actual target values. Common loss functions include mean squared error for regression and cross-entropy for classification.

2. **Gradient Computation:** The gradients of the loss function with respect to the model parameters are computed using the chain rule. This is done by calculating how changes in each parameter affect the loss.

3. **Parameter Updates:** Optimization algorithms, such as gradient descent, use the computed gradients to adjust the model's parameters to minimize the loss. The parameter update is typically done using the following equation:
```
 New Parameter Value=Old Parameter Value−Learning Rate×Gradient
```
represents the parameter update rule used in the context of optimization algorithms, such as gradient descent. Let's break down this formula:

New Parameter Value: This is the updated value of a specific parameter (e.g., weight or bias) in a neural network that the optimization algorithm is trying to optimize.

Old Parameter Value: This is the current value of the parameter that you want to update.

Learning Rate: The learning rate is a hyperparameter that controls the size of the update step. It determines how much the parameter values should change in each iteration of the optimization process. A larger learning rate results in larger steps, potentially speeding up convergence but risking instability, while a smaller learning rate leads to smaller steps, which may improve stability but slow down convergence.

Gradient: The gradient represents the vector of partial derivatives of the loss function with respect to the parameter being updated. In the context of neural network training, this gradient is calculated during the backward propagation phase (backpropagation). It indicates the direction in which the parameter should be adjusted to reduce the loss. The gradient points to the direction of the steepest ascent in the loss landscape, so subtracting it from the old parameter value moves the parameter in the direction of steepest descent (toward a lower loss value).

This process of forward and backward propagation continues for a set number of iterations or until a convergence criterion is met.

## Training Objective

The objective of training is to minimize the loss function, which quantifies the model's error. As the model iteratively updates its parameters using gradients, it gradually improves its ability to make accurate predictions on the training data.

