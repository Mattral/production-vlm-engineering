"""
Uses
1.Feature Learning and Dimensionality Reduction:

Autoencoders are often used for feature learning, where the encoder part of the network learns a compact representation (encoding) of the input data. This can be useful for reducing the dimensionality of the data, capturing important features, and removing noise.

2.Anomaly Detection:

Autoencoders can be employed for anomaly detection. If the model is trained on normal data, it can reconstruct normal instances well. Anomalies, which are not well-reconstructed, can be detected.

3.Image Denoising:

By training an autoencoder to denoise images, it can learn to remove noise from corrupted images, thus providing a cleaner version.

4.Generative Modeling:

Variational Autoencoders (a type of autoencoder) are used for generative modeling. They can generate new samples similar to the training data.

5.Transfer Learning:

Autoencoders trained on one dataset can be used as a pre-training step for other tasks, providing a useful initialization for the weights.

6.Regression and Time Series:

While autoencoders are not the typical choice for regression or time series prediction tasks, the encoded features can be used as inputs to subsequent regression models.

7.Natural Language Processing (NLP):

In NLP, autoencoders can be applied to learn meaningful representations of text data, useful for tasks like text summarization, document clustering, and sentiment analysis.
"""

import numpy as np
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(self, input_size, hidden_size):
        self.weights_encoder = np.random.randn(hidden_size, input_size)
        self.bias_encoder = np.zeros((hidden_size, 1))
        
        self.weights_decoder = np.random.randn(input_size, hidden_size)
        self.bias_decoder = np.zeros((input_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def encode(self, x):
        z = np.dot(self.weights_encoder, x) + self.bias_encoder
        return self.sigmoid(z)

    def decode(self, z):
        x_hat = np.dot(self.weights_decoder, z) + self.bias_decoder
        return self.sigmoid(x_hat)

    def train(self, x, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            z = self.encode(x)
            x_hat = self.decode(z)

            # Compute loss (mean squared error)
            loss = np.mean((x_hat - x)**2)

            # Backward pass (gradient descent)
            delta_x_hat = 2 * (x_hat - x)
            delta_z = np.dot(self.weights_decoder.T, delta_x_hat) * (z * (1 - z))

            # Update weights and biases
            self.weights_encoder -= learning_rate * np.dot(delta_z, x.T)
            self.bias_encoder -= learning_rate * np.sum(delta_z, axis=1, keepdims=True)
            self.weights_decoder -= learning_rate * np.dot(delta_x_hat, z.T)
            self.bias_decoder -= learning_rate * np.sum(delta_x_hat, axis=1, keepdims=True)

            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

# Generate random data (e.g., 100 samples of 10-dimensional vectors)
data = np.random.randn(10, 100)

# Create an autoencoder with input size 10 and hidden size 5
autoencoder = Autoencoder(input_size=10, hidden_size=5)

# Train the autoencoder
autoencoder.train(data, learning_rate=0.01, epochs=1000)

# Encode and decode the data
encoded_data = autoencoder.encode(data)
decoded_data = autoencoder.decode(encoded_data)

# Plot the original and reconstructed data
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(data[0, :], data[1, :], color='blue', label='Original Data')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(decoded_data[0, :], decoded_data[1, :], color='red', label='Reconstructed Data')
plt.title('Reconstructed Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.show()
