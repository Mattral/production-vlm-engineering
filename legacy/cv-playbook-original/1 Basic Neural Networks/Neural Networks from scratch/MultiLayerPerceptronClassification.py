import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the Multi-Layer Perceptron.

        Parameters:
        - input_size: int
            Number of input features.
        - hidden_size: int
            Number of neurons in the hidden layer.
        - output_size: int
            Number of output neurons.
        """
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_output = np.random.rand(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, X):
        # Forward pass through the network
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output_probabilities = self.softmax(output_input)
        return hidden_input, hidden_output, output_input, output_probabilities

    def backward(self, X, y, hidden_input, hidden_output, output_input, output_probabilities, learning_rate=0.01):
        # Backward pass and weight updates
        output_error = output_probabilities - y
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * hidden_output * (1 - hidden_output)

        self.weights_hidden_output -= learning_rate * np.dot(hidden_output.T, output_error)
        self.bias_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
        self.weights_input_hidden -= learning_rate * np.dot(X.T, hidden_error)
        self.bias_hidden -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            # Forward pass
            hidden_input, hidden_output, output_input, output_probabilities = self.forward(X)

            # Cross-entropy loss
            loss = -np.sum(y * np.log(output_probabilities)) / len(X)

            # Backward pass and weight updates
            self.backward(X, y, hidden_input, hidden_output, output_input, output_probabilities, learning_rate)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Loss = {loss}')

    def predict(self, X):
        _, _, _, output_probabilities = self.forward(X)
        return np.argmax(output_probabilities, axis=1)
