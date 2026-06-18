'''
Below is a basic implementation of a self-attention mechanism:

This example assumes a simple scenario with random input matrices for query,
key, and value. In a real-world scenario, these matrices would represent embeddings or
representations of the input sequence at different positions.
'''


import numpy as np

def scaled_dot_product_attention(query, key, value):
    """
    Scaled Dot-Product Self-Attention mechanism.

    Parameters:
    - query (numpy.ndarray): Query matrix (shape: [sequence_length, d_model]).
    - key (numpy.ndarray): Key matrix (shape: [sequence_length, d_model]).
    - value (numpy.ndarray): Value matrix (shape: [sequence_length, d_model]).

    Returns:
    numpy.ndarray: Context matrix (shape: [sequence_length, d_model]).
    """
    d_model = query.shape[-1]  # Dimension of query/key/value vectors
    scores = np.dot(query, key.T) / np.sqrt(d_model)  # Dot product with scaling
    attention_weights = softmax(scores, axis=-1)  # Apply softmax to get attention weights
    context_matrix = np.dot(attention_weights, value)  # Weighted sum to get context matrix
    return context_matrix

def softmax(x, axis=-1):
    """
    Softmax function.

    Parameters:
    - x (numpy.ndarray): Input array.
    - axis (int): Axis along which the softmax is computed.

    Returns:
    numpy.ndarray: Softmax output.
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Example Usage
sequence_length = 5
d_model = 3

# Random input matrices for query, key, and value
query = np.random.randn(sequence_length, d_model)
key = np.random.randn(sequence_length, d_model)
value = np.random.randn(sequence_length, d_model)

# Self-Attention
context_matrix = scaled_dot_product_attention(query, key, value)

print("Query Matrix:")
print(query)
print("\nKey Matrix:")
print(key)
print("\nValue Matrix:")
print(value)
print("\nContext Matrix (Output of Self-Attention):")
print(context_matrix)
