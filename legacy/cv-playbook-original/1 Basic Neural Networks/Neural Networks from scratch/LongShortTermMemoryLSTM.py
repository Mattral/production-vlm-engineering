import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LSTMCell:
    """
    LSTM Cell implementation.

    Args:
    - input_size (int): Size of the input vector.
    - hidden_size (int): Size of the hidden state.

    Attributes:
    - input_size (int): Size of the input vector.
    - hidden_size (int): Size of the hidden state.
    - W_ih (ndarray): Weight matrix for input.
    - b_ih (ndarray): Bias vector for input.
    - h_t (ndarray): Hidden state.
    - c_t (ndarray): Cell state.
    - f_t, i_t, g_t, o_t (ndarray): Forget, input, cell, and output gates.

    Methods:
    - sigmoid(x): Sigmoid activation function.
    - tanh(x): Hyperbolic tangent activation function.
    - forward(x_t): Forward pass through the LSTM cell.
    """

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters
        self.W_ih = np.random.randn(4 * hidden_size, input_size + hidden_size)
        self.b_ih = np.zeros((4 * hidden_size, 1))
        self.h_t = np.zeros((hidden_size, 1))
        self.c_t = np.zeros((hidden_size, 1))
        self.f_t = None
        self.i_t = None
        self.g_t = None
        self.o_t = None

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Args:
        - x (ndarray): Input.

        Returns:
        - ndarray: Sigmoid output.
        """
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Hyperbolic tangent activation function.

        Args:
        - x (ndarray): Input.

        Returns:
        - ndarray: Tanh output.
        """
        return np.tanh(x)

    def forward(self, x_t):
        """
        Forward pass through the LSTM cell.

        Args:
        - x_t (ndarray): Input vector for the current timestep.

        Returns:
        - ndarray: Hidden state for the current timestep.
        """
        # Concatenate the input and hidden states for efficiency
        x_concat = np.concatenate((x_t, self.h_t))

        # Compute activations
        a_ih = np.dot(self.W_ih, x_concat) + self.b_ih

        # Slice activations for the gates
        a_f_t, a_i_t, a_g_t, a_o_t = np.split(a_ih, 4)

        # Apply activation functions
        self.f_t = self.sigmoid(a_f_t)
        self.i_t = self.sigmoid(a_i_t)
        self.g_t = self.tanh(a_g_t)
        self.o_t = self.sigmoid(a_o_t)

        # Update cell state and hidden state
        self.c_t = self.f_t * self.c_t + self.i_t * self.g_t
        self.h_t = self.o_t * self.tanh(self.c_t)

        return self.h_t


# Example usage with sequential data
input_size = 1
hidden_size = 2
lstm_cell = LSTMCell(input_size, hidden_size)

# Create a sequence of input vectors
sequence_length = 10
data_sequence = np.random.randn(input_size, sequence_length)

# Forward pass through the LSTM cell for each timestep
hidden_states = []
for t in range(sequence_length):
    x_t = data_sequence[:, [t]]
    h_t = lstm_cell.forward(x_t)
    hidden_states.append(h_t.flatten())

# Plot the hidden states over time
plt.figure()
for i, hidden_state in enumerate(hidden_states):
    plt.scatter(np.full_like(hidden_state, i), hidden_state, label=f'Timestep {i}')

plt.xlabel('Timestep')
plt.ylabel('Hidden State Value')
plt.title('LSTM Cell Hidden States Over Time')
plt.legend()
plt.show()

# Load data from CSV file
file_path = 'Microsoft_Stock.csv'
df = pd.read_csv(file_path)

# Choose a target variable (e.g., closing prices)
target_variable = 'Close'
data_sequence = df[target_variable].values.reshape(1, -1)

# Normalize the data
data_sequence = (data_sequence - np.mean(data_sequence)) / np.std(data_sequence)

# LSTMCell class definition here...

# Example usage with time series data
input_size = 1
hidden_size = 2
lstm_cell = LSTMCell(input_size, hidden_size)

# Forward pass through the LSTM cell for each timestep
hidden_states = []
for t in range(data_sequence.shape[1]):
    x_t = data_sequence[:, [t]]
    h_t = lstm_cell.forward(x_t)
    hidden_states.append(h_t.flatten())

# Plot the hidden states over time
plt.figure()
for i, hidden_state in enumerate(hidden_states):
    plt.scatter(np.full_like(hidden_state, i), hidden_state, label=f'Timestep {i}')

plt.xlabel('Timestep')
plt.ylabel('Hidden State Value')
plt.title('LSTM Cell Hidden States Over Time (Microsoft Stock)')
plt.legend()
plt.show()
