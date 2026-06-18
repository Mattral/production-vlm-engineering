import numpy as np
import matplotlib.pyplot as plt

def crelu(z):
    """
    Complex Rectified Linear Unit (CReLU) activation function.

    Parameters:
    - z (array): Input array of complex numbers.

    Returns:
    - array: Output array after applying the CReLU activation.
    """
    return np.where(z.real > 0, z, 0)

def complex_sgd(w, grad, learning_rate):
    """
    Complex-valued Stochastic Gradient Descent (SGD) update rule.

    Parameters:
    - w (array): Complex-valued weights.
    - grad (array): Gradient of the weights.
    - learning_rate (float): Learning rate.

    Returns:
    - array: Updated complex-valued weights.
    """
    return w - learning_rate * grad

def initialize_weights(input_size, hidden_size, output_size):
    """
    Initialize complex-valued weights and biases for a fully connected network.

    Parameters:
    - input_size (int): Number of input features.
    - hidden_size (int): Number of units in the hidden layer.
    - output_size (int): Number of output units.

    Returns:
    - tuple: Complex-valued weights and biases for each layer.
    """
    w1 = np.random.normal(0, 1, (input_size, hidden_size)) + 1j * np.random.normal(0, 1, (input_size, hidden_size))
    b1 = np.random.normal(0, 1, (1, hidden_size)) + 1j * np.random.normal(0, 1, (1, hidden_size))
    
    w2 = np.random.normal(0, 1, (hidden_size, output_size)) + 1j * np.random.normal(0, 1, (hidden_size, output_size))
    b2 = np.random.normal(0, 1, (1, output_size)) + 1j * np.random.normal(0, 1, (1, output_size))
    
    return w1, b1, w2, b2


def forward_pass(x, w1, b1, w2, b2):
    """
    Perform a forward pass through the complex-valued neural network.

    Parameters:
    - x (array): Input data.
    - w1, b1, w2, b2 (tuple): Complex-valued weights and biases for each layer.

    Returns:
    - array: Output of the neural network.
    """
    z1 = np.dot(x, w1) + b1
    a1 = crelu(z1)
    
    z2 = np.dot(a1, w2) + b2
    a2 = crelu(z2)
    
    return a2

def backward_pass(x, y_true, a2, w1, b1, a1, w2, b2, learning_rate):
    """
    Perform a backward pass through the complex-valued neural network to update weights.

    Parameters:
    - x (array): Input data.
    - y_true (array): True labels.
    - a2 (array): Output of the neural network.
    - w1, b1, a1, w2, b2 (tuple): Complex-valued weights and biases for each layer and activation of the first layer.
    - learning_rate (float): Learning rate.

    Returns:
    - tuple: Updated complex-valued weights and biases for each layer.
    """
    # Define z1 explicitly
    z1 = np.dot(x, w1) + b1
    
    delta = a2 - y_true
    
    # Gradient for the second layer
    grad_w2 = np.dot(a1.T, delta)
    grad_b2 = np.sum(delta, axis=0, keepdims=True)
    
    # Gradient for the first layer
    delta_a1 = np.dot(delta, w2.T)
    delta_a1[z1.real <= 0] = 0  # Derivative of CReLU
    grad_w1 = np.dot(x.T, delta_a1)
    grad_b1 = np.sum(delta_a1, axis=0, keepdims=True)
    
    # Update weights using complex SGD
    w2 = complex_sgd(w2, grad_w2, learning_rate)
    b2 = complex_sgd(b2, grad_b2, learning_rate)
    w1 = complex_sgd(w1, grad_w1, learning_rate)
    b1 = complex_sgd(b1, grad_b1, learning_rate)
    
    return w1, b1, w2, b2




def visualize_complex_weights(w, title):
    """
    Visualize complex weights as angles in HSV color space.

    Parameters:
    - w (array): Complex-valued weights.
    - title (str): Title for the plot.
    """
    plt.figure()
    plt.imshow(np.angle(w), cmap='hsv', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()

# Generate some complex-valued input data
x = np.array([[1.0 + 2.0j, -3.0 - 1.0j]])

# Ground truth for demonstration purposes
y_true = np.array([[0.5 + 0.5j]])

# Network parameters
input_size = x.shape[1]
hidden_size = 10
output_size = y_true.shape[1]

# Initialize complex-valued weights
w1, b1, w2, b2 = initialize_weights(input_size, hidden_size, output_size)

# Training parameters
learning_rate = 0.01
num_epochs = 1000

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    a2 = forward_pass(x, w1, b1, w2, b2)
    
    # Backward pass
    w1, b1, w2, b2 = backward_pass(x, y_true, a2, w1, b1, crelu(np.dot(x, w1) + b1), w2, b2, learning_rate)

    
    # Visualize complex weights every 100 epochs
    if (epoch + 1) % 100 == 0:
        visualize_complex_weights(w1, f'Weights (Layer 1) - Epoch {epoch+1}')
        visualize_complex_weights(w2, f'Weights (Layer 2) - Epoch {epoch+1}')

# Print final output
print("Final Output:")
print(a2)
