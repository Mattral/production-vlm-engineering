import numpy as np
import matplotlib.pyplot as plt


class MultiLayerPerceptronRegression:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the Multi-Layer Perceptron for regression.

        Parameters:
        - input_size: int
            Number of input features.
        - hidden_size: int
            Number of neurons in the hidden layer.
        - output_size: int
            Number of output neurons (1 for regression).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases for the hidden layer
        self.weights_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.rand(1, self.hidden_size)

        # Initialize weights and biases for the output layer
        self.weights_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.random.rand(1, self.output_size)

    def linear_activation(self, x):
        """
        Linear activation function.

        Parameters:
        - x: numpy array
            Input to the linear activation function.

        Returns:
        - numpy array
            Output of the linear activation function.
        """
        return x

    def mean_squared_error(self, y_true, y_pred):
        """
        Compute mean squared error loss.

        Parameters:
        - y_true: numpy array
            True labels.
        - y_pred: numpy array
            Predicted labels.

        Returns:
        - float
            Mean squared error loss.
        """
        return np.mean((y_true - y_pred) ** 2)

    def forward_pass(self, X):
        """
        Perform the forward pass through the network.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.

        Returns:
        - numpy array
            Predicted output.
        """
        # Hidden layer
        hidden_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        hidden_output = self.linear_activation(hidden_input)

        # Output layer
        output_input = np.dot(hidden_output, self.weights_output) + self.bias_output
        output = self.linear_activation(output_input)

        return output

    def backward_pass(self, X, y, output):
        """
        Perform the backward pass to update weights and biases.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
        - y: numpy array
            True labels.
        - output: numpy array
            Predicted output.
        """
        # Output layer gradient
        output_error = y - output
        output_delta = output_error

        # Hidden layer gradient
        hidden_error = output_delta.dot(self.weights_output.T)
        hidden_delta = hidden_error

        # Update weights and biases
        self.weights_output += X.T.dot(output_delta)
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True)
        self.weights_hidden += X.T.dot(hidden_delta)
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, learning_rate=0.01, epochs=1000):
        """
        Train the Multi-Layer Perceptron for regression.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
        - y: numpy array
            True labels.
        - learning_rate: float, optional (default=0.01)
            Learning rate for gradient descent.
        - epochs: int, optional (default=1000)
            Number of training epochs.
        """
        for epoch in range(epochs):
            # Forward pass
            output = self.forward_pass(X)

            # Compute mean squared error loss
            loss = self.mean_squared_error(y, output)

            # Backward pass
            self.backward_pass(X, y, output)

            # Print details every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss = {loss}')

        # Print final details after training
        print(f'Final Epoch: Loss = {loss}')


# Generate synthetic data for regression
np.random.seed(42)
X = np.random.randn(200, 1)
y = 3 * X + 2 + 0.5 * np.random.randn(200, 1)

if np.isnan(my_array).any():
    print("There are nan values in the array.")
    
# Create an instance of MultiLayerPerceptronRegression
mlp_regression = MultiLayerPerceptronRegression(input_size=1, hidden_size=3, output_size=1)

# Train the Multi-Layer Perceptron
mlp_regression.train(X, y, learning_rate=0.01, epochs=1000)

# Make predictions on the same data for visualization
predictions = mlp_regression.forward_pass(X)

# Plot the original data and the predictions
plt.scatter(X, y, label='Original Data')
plt.plot(X, predictions, color='red', label='Predictions')
plt.title('Regression with Multi-Layer Perceptron')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
