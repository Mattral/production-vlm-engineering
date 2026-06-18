import numpy as np
import matplotlib.pyplot as plt

class SingleLayerPerceptronRegression:
    def __init__(self, input_size):
        """
        Initialize the Single Layer Perceptron for Regression.

        Parameters:
        - input_size: int
            Number of input features.
        """
        # Initialize random weights and bias
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)

    def linear_activation(self, x):
        """
        Linear activation function (identity function).

        Parameters:
        - x: numpy array
            Input to the activation function.

        Returns:
        - numpy array
            Output of the activation function.
        """
        return x

    def predict(self, X):
        """
        Make predictions using the trained perceptron.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.

        Returns:
        - numpy array
            Predicted values.
        """
        # Calculate the weighted sum using linear activation
        z = np.dot(X, self.weights) + self.bias
        predictions = self.linear_activation(z)
        return predictions

    def train(self, X, y, learning_rate=0.01, epochs=1000, epsilon=1e-8):
        """
        Train the Single Layer Perceptron for Regression.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
        - y: numpy array, shape (n_samples,)
            Target values.
        - learning_rate: float, optional (default=0.01)
            Learning rate for gradient descent.
        - epochs: int, optional (default=1000)
            Number of training epochs.
        - epsilon: float, optional (default=1e-8)
            Small value for numerical stability.
        """
        for epoch in range(epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self.linear_activation(z)

            # Compute the mean squared error loss with epsilon for numerical stability
            loss = np.mean((predictions - y)**2)

            # Backward pass
            dw = np.dot(X.T, predictions - y)
            db = np.sum(predictions - y)

            # Update weights and bias using gradient descent
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            # Print details every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss = {loss}')

        # Print final details after training
        print(f'Final Epoch: Loss = {loss}')
        print(f'Final Weights: {self.weights}')
        print(f'Final Bias: {self.bias}')


# Generate synthetic data for regression
np.random.seed(42)
X = np.random.randn(200, 1)
y = 2 * X.squeeze() + 1 + 0.5 * np.random.randn(200)  # Linear relationship with some noise

# Train the Single-Layer Perceptron for Regression
slp_regression = SingleLayerPerceptronRegression(input_size=1)
slp_regression.train(X, y, learning_rate=0.01, epochs=500)
