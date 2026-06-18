'''
Single Layer Perceptron (SLP): It's the simplest form of a neural network with a single layer of neurons. The perceptron makes predictions based on a weighted sum of input features and applies an activation function (sigmoid in this case).

Initialization: We initialize the weights and bias randomly.

Sigmoid Activation Function: It squashes the weighted sum into the range [0, 1].

Predict Method: It calculates the predictions based on the current weights and bias.

Training: It uses gradient descent to update weights and bias in the direction that minimizes the binary cross-entropy loss.

Plot Decision Boundary: It visualizes the decision boundary during training to see how the model learns.

Example Usage: It generates synthetic data and trains the perceptron on it.


'''

import numpy as np
import matplotlib.pyplot as plt

class SingleLayerPerceptron:
    def __init__(self, input_size):
        """
        Initialize the Single Layer Perceptron.

        Parameters:
        - input_size: int
            Number of input features.
        """
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Parameters:
        - x: numpy array
            Input to the sigmoid function.

        Returns:
        - numpy array
            Output of the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        """
        Make predictions using the trained perceptron.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.

        Returns:
        - numpy array
            Predicted labels (0 or 1).
        """
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return (predictions > 0.5).astype(int)

    def train(self, X, y, learning_rate=0.01, epochs=1000, epsilon=1e-8):
        """
        Train the Single Layer Perceptron.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
        - y: numpy array, shape (n_samples,)
            Binary labels (0 or 1).
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
            predictions = self.sigmoid(z)

            # Compute the binary cross-entropy loss with epsilon for numerical stability
            loss = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))

            # Backward pass
            dw = np.dot(X.T, predictions - y)
            db = np.sum(predictions - y)

            # Update weights and bias
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            # Print details every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss = {loss}')

                # You can print additional information if needed
                # For example, you can print weights and bias:
                print(f'  Weights: {self.weights}')
                print(f'  Bias: {self.bias}')

                self.plot_decision_boundary(X, y, epoch)

        # Print final details after training
        print(f'Final Epoch: Loss = {loss}')
        print(f'Final Weights: {self.weights}')
        print(f'Final Bias: {self.bias}')


    def plot_decision_boundary(self, X, y, epoch):
        """
        Plot the decision boundary of the perceptron.

        Parameters:
        - X: numpy array, shape (n_samples, n_features)
            Input features.
        - y: numpy array, shape (n_samples,)
            Binary labels (0 or 1).
        - epoch: int
            Current training epoch.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
        plt.title(f'Decision Boundary - Epoch {epoch}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contour(xx, yy, Z, levels=[0.5], colors='black')  # Use contour instead of contourf
        plt.show()


# Generate synthetic data for binary classification
np.random.seed(42)
X = np.random.randn(200, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Train the Single-Layer Perceptron
slp = SingleLayerPerceptron(input_size=2)
slp.train(X, y, learning_rate=0.1, epochs=500)
