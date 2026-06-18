'''
Simple Bayesian Neural Network (BNN) with one hidden layer
This architecture is relatively simple and may not be suitable for complex tasks. 
The choice of architecture, including the number of layers and units, is often problem-specific. 
In practice, more complex architectures with multiple hidden layers and different activation functions might be used for more challenging tasks
'''


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train, y_train = torch.from_numpy(X_train).float(), torch.from_numpy(y_train)
X_test, y_test = torch.from_numpy(X_test).float(), torch.from_numpy(y_test)

# Define Bayesian Neural Network model
class BayesianNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Bayesian Neural Network model with Gaussian priors and posteriors.

        Parameters:
        - input_size (int): Number of features in the input.
        - hidden_size (int): Number of units in the hidden layer.
        - output_size (int): Number of units in the output layer.
        """
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Define prior distributions for weights
        self.prior_mean = 0
        self.prior_std = 1

    def forward(self, x):
        """
        Forward pass of the Bayesian Neural Network.

        Parameters:
        - x (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor.
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def calculate_likelihood(self, y_pred, y_true):
        """
        Calculate the likelihood of the observed data given the predicted data.

        Parameters:
        - y_pred (Tensor): Predicted data.
        - y_true (Tensor): True data.

        Returns:
        - Tensor: Likelihood.
        """
        # Assume Gaussian likelihood
        likelihood = torch.distributions.Normal(y_pred, 1e-3).log_prob(y_true).sum()
        return likelihood

    def calculate_prior(self):
        """
        Calculate the prior distribution of the weights.

        Returns:
        - Tensor: Prior distribution.
        """
        # Assume Gaussian prior for weights
        prior = torch.distributions.Normal(0, 1).log_prob(self.fc1.weight).sum() + \
                torch.distributions.Normal(0, 1).log_prob(self.fc2.weight).sum()
        return prior

    def calculate_posterior(self):
        """
        Calculate the posterior distribution of the weights.

        Returns:
        - Tensor: Posterior distribution.
        """
        # Assume Gaussian posterior for weights
        posterior = torch.distributions.Normal(self.fc1.weight, 1e-3).log_prob(self.fc1.weight).sum() + \
                     torch.distributions.Normal(self.fc2.weight, 1e-3).log_prob(self.fc2.weight).sum()
        return posterior

# Training function
def train_bayesian_nn(model, X, y, optimizer, num_samples=10):
    """
    Train the Bayesian Neural Network using the Evidence Lower Bound (ELBO) loss.

    Parameters:
    - model (BayesianNN): Bayesian Neural Network model.
    - X (Tensor): Input data.
    - y (Tensor): Target data.
    - optimizer (torch.optim.Optimizer): PyTorch optimizer.
    - num_samples (int): Number of samples for stochastic gradient estimation.

    Returns:
    - float: Total loss.
    """
    model.train()
    optimizer.zero_grad()

    total_loss = 0
    for _ in range(num_samples):
        y_pred = model(X)
        likelihood = model.calculate_likelihood(y_pred, y)
        prior = model.calculate_prior()
        posterior = model.calculate_posterior()

        # ELBO (Evidence Lower Bound)
        loss = -(likelihood - (1/num_samples) * (posterior - prior))
        total_loss += loss.item()

    total_loss /= num_samples
    loss.backward()
    optimizer.step()

    return total_loss

# Prediction function with uncertainty estimates
def predict_with_uncertainty(model, X, num_samples=100):
    """
    Make predictions using the Bayesian Neural Network with uncertainty estimates.

    Parameters:
    - model (BayesianNN): Bayesian Neural Network model.
    - X (Tensor): Input data.
    - num_samples (int): Number of samples for uncertainty estimation.

    Returns:
    - Tuple: Mean prediction and standard deviation of predictions.
    """
    model.eval()
    predictions = []

    for _ in range(num_samples):
        y_pred = model(X)
        predictions.append(y_pred.detach().numpy())

    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    std_prediction = np.std(predictions, axis=0)

    return mean_prediction, std_prediction

# Visualize predictions with uncertainty
def visualize_predictions(X, y_true, mean_prediction, std_prediction):
    """
    Visualize predictions with uncertainty.

    Parameters:
    - X (array): Input data.
    - y_true (array): True target data.
    - mean_prediction (array): Mean predictions.
    - std_prediction (array): Standard deviation of predictions.
    """
    # Select a specific feature for the x-axis (assuming the first feature)
    x_feature = 0

    plt.scatter(X[:, x_feature], y_true, label='True Data', color='blue')
    plt.plot(X[:, x_feature], mean_prediction, label='Mean Prediction', color='red')
    plt.fill_between(X[:, x_feature], mean_prediction - 2 * std_prediction, mean_prediction + 2 * std_prediction, color='orange', alpha=0.3, label='Uncertainty (2 std)')
    plt.title('Bayesian Neural Network Predictions with Uncertainty')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.show()

# Initialize Bayesian Neural Network model
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1
bayesian_nn_model = BayesianNN(input_size, hidden_size, output_size)

# Define optimizer
optimizer = optim.Adam(bayesian_nn_model.parameters(), lr=0.01)

# Train Bayesian Neural Network
num_epochs = 500
for epoch in range(num_epochs):
    total_loss = train_bayesian_nn(bayesian_nn_model, X_train, y_train, optimizer)
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')

# Test Bayesian Neural Network on test data
mean_pred, std_pred = predict_with_uncertainty(bayesian_nn_model, X_test)

# Print test data and predictions
print("Test Data:")
print(X_test.numpy())
print("True Labels:")
print(y_test.numpy())
print("Mean Predictions:")
print(mean_pred.squeeze())
print("Standard Deviation of Predictions:")
print(std_pred.squeeze())

# Visualize predictions with uncertainty
visualize_predictions(X_test.numpy(), y_test.numpy(), mean_pred.squeeze(), std_pred.squeeze())
