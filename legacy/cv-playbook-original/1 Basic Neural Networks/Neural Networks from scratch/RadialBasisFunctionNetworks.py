'''
Radial Basis Function (RBF) Networks are a type of neural network that uses radial basis functions as activation functions.

Architecture:

-Input Layer: Takes input features.
-Hidden Layer: Comprises nodes, each associated with a radial basis function. These functions measure the distance between the input and a center (a point in the input space).
-Output Layer: Combines the outputs of the hidden layer to produce the final output.
______________________________
Activation Function (Radial Basis Function):

-The radial basis function is typically a Gaussian function,
which calculates the distance between the input and a center.
-The output of the radial basis function is highest when the input is close
to the center and decreases as the distance increases.
____________________________
Usage:

1. Function Approximation: RBF networks are often used for function approximation
tasks where the relationship between inputs and outputs is not explicitly known.

2. Pattern Recognition: They can be applied to pattern recognition tasks,
such as classification problems.
3. Interpolation and Extrapolation: RBF networks can be used for
interpolation(predicting values within the training data)
and extrapolation (predicting values outside the training data).
_______________
Advantages:

Universal Approximator: RBF networks are known to be universal approximators, meaning they can approximate any continuous function to arbitrary precision given a sufficient number of radial basis functions.
Localized Learning: The use of radial basis functions allows for localized learning, where each function focuses on a specific region of the input space.

_____________
Challenges:

Choice of Centers: The choice of centers for the radial basis functions can significantly impact the performance of the network.
Computational Complexity: Training RBF networks can be computationally intensive, especially when determining the centers.

'''


import numpy as np
import matplotlib.pyplot as plt

class RadialBasisFunctionNetwork:
    def __init__(self, num_rbf, learning_rate=0.1):
        """
        Initialize a Radial Basis Function Network.

        Parameters:
        - num_rbf (int): Number of radial basis functions.
        - learning_rate (float): Learning rate for weight updates.
        """
        self.num_rbf = num_rbf
        self.centers = None
        self.weights = None
        self.learning_rate = learning_rate

    def radial_basis_function(self, x, c, sigma=1):
        """
        Radial Basis Function (Gaussian) activation.

        Parameters:
        - x (numpy.ndarray): Input vector.
        - c (numpy.ndarray): Center vector.
        - sigma (float): Width parameter for the Gaussian.

        Returns:
        float: RBF activation.
        """
        return np.exp(-(np.linalg.norm(x - c) ** 2) / (2 * sigma ** 2))

    def initialize_centers(self, data):
        """
        Initialize RBF centers using k-means clustering.

        Parameters:
        - data (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Initialized RBF centers.
        """
        idx = np.random.choice(len(data), self.num_rbf, replace=False)
        centers = data[idx]
        return centers

    def train(self, data, targets, num_epochs=100):
        """
        Train the RBF network.

        Parameters:
        - data (numpy.ndarray): Input data.
        - targets (numpy.ndarray): Target outputs.
        - num_epochs (int): Number of training epochs.

        Returns:
        None
        """
        self.centers = self.initialize_centers(data)
        self.weights = np.random.rand(self.num_rbf)

        for epoch in range(num_epochs):
            for i in range(len(data)):
                rbf_activations = np.array([self.radial_basis_function(data[i], c) for c in self.centers])
                output = np.dot(rbf_activations, self.weights)

                error = targets[i] - output
                self.weights += self.learning_rate * error * rbf_activations

    def predict(self, data):
        """
        Make predictions using the trained RBF network.

        Parameters:
        - data (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Predicted outputs.
        """
        predictions = []
        for i in range(len(data)):
            rbf_activations = np.array([self.radial_basis_function(data[i], c) for c in self.centers])
            output = np.dot(rbf_activations, self.weights)
            predictions.append(output)
        return np.array(predictions)

def visualize_samples(samples, title):
    """
    Visualize generated samples from the Boltzmann Machine.

    Parameters:
    - samples (numpy.ndarray): Generated samples.
    - title (str): Title for the visualization.

    Returns:
    None
    """
    num_samples = samples.shape[1]
    sample_size = samples.shape[0]

    plt.figure(figsize=(15, 2 * num_samples))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.stem(range(sample_size), samples[:, i])
        plt.title(f'Sample {i + 1}')

    plt.suptitle(title)
    plt.show()

# Example Usage
num_visible_nodes = 5
num_hidden_nodes = 3
rbf_network = RadialBasisFunctionNetwork(num_visible_nodes)

# Generate synthetic data
np.random.seed(42)
X = np.sort(5 * np.random.rand(20, 1), axis=0)
y = np.sin(X).ravel()

# Train the RBF network
rbf_network.train(X, y, num_epochs=1000)

# Make predictions
X_pred = np.arange(0, 5, 0.1).reshape(-1, 1)
y_pred = rbf_network.predict(X_pred)

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(rbf_network.centers, np.zeros_like(rbf_network.centers), color='blue', marker='o', label='RBF Centers')
plt.plot(X, y, 'r-', label='Original Function')
plt.plot(X_pred, y_pred, 'g-', label='RBF Network Approximation')
plt.legend()
plt.title('Radial Basis Function Network')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()

