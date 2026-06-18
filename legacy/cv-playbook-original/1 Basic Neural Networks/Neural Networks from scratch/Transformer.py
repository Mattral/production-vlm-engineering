'''
The code provided is a simplified and basic implementation of a Transformer model,
a type of neural network architecture that has been highly successful in natural language processing (NLP)
and other sequence-to-sequence tasks. The implementation is kept minimal
for educational purposes and lacks many optimizations and features found in full-scale deep learning libraries.
'''


import numpy as np

def gelu(x):
    """
    gelu: GELU (Gaussian Error Linear Unit) activation function.

    Parameters:
    - x (numpy.ndarray): Input.

    Returns:
    numpy.ndarray: Output after applying GELU.
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x):
    """
    Softmax activation function.

    Parameters:
    - x (numpy.ndarray): Input.

    Returns:
    numpy.ndarray: Output after applying softmax.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, epsilon=1e-5):
    """
    Layer normalization.

    Parameters:
    - x (numpy.ndarray): Input.
    - epsilon (float, optional): Small value for numerical stability.

    Returns:
    numpy.ndarray: Output after layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    std_dev = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std_dev + epsilon)

def positional_encoding(max_len, d_model):
    """
     Generates positional encoding for a sequence, providing information about the position of tokens.
     It is used to incorporate the sequential order into the model.

    Parameters:
    - max_len (int): Maximum length of the sequence.
    - d_model (int): Dimensionality of the model.

    Returns:
    numpy.ndarray: Positional encoding matrix of shape (max_len, d_model).
    """
    pos_enc = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            angle = pos / np.power(10000, i / d_model)
            pos_enc[pos, i] = np.sin(angle)
            pos_enc[pos, i + 1] = np.cos(angle)
    return pos_enc

def scaled_dot_product_attention(Q, K, V):
    """
    Performs scaled dot-product attention, a key component in the self-attention mechanism.

    Parameters:
    - Q (numpy.ndarray): Queries.
    - K (numpy.ndarray): Keys.
    - V (numpy.ndarray): Values.

    Returns:
    numpy.ndarray: Attended values.
    """
    scale_factor = np.sqrt(Q.shape[-1])
    attention_scores = np.matmul(Q, K.T) / scale_factor
    attention_weights = softmax(attention_scores)
    attended_values = np.matmul(attention_weights, V)
    return attended_values

def multi_head_attention(Q, K, V, num_heads):
    """
    Combines multiple heads of self-attention to capture different aspects of the input sequence.

    Parameters:
    - Q (numpy.ndarray): Queries.
    - K (numpy.ndarray): Keys.
    - V (numpy.ndarray): Values.
    - num_heads (int): Number of attention heads.

    Returns:
    numpy.ndarray: Attended values.
    """
    head_dim = Q.shape[-1] // num_heads
    Q_split = np.split(Q, num_heads, axis=-1)
    K_split = np.split(K, num_heads, axis=-1)
    V_split = np.split(V, num_heads, axis=-1)
    attended_values_split = [scaled_dot_product_attention(Q_h, K_h, V_h) for Q_h, K_h, V_h in zip(Q_split, K_split, V_split)]
    return np.concatenate(attended_values_split, axis=-1)

def feedforward(x, d_model, d_ff):
    """
    Feedforward layer.

    Parameters:
    - x (numpy.ndarray): Input.
    - d_model (int): Dimensionality of the model.
    - d_ff (int): Dimensionality of the feedforward layer.

    Returns:
    numpy.ndarray: Output after the feedforward layer.
    """
    linear1 = np.random.randn(d_model, d_ff)
    linear2 = np.random.randn(d_ff, d_model)
    return gelu(np.dot(np.maximum(0, np.dot(x, linear1)), linear2))

def transformer_block(x, d_model, num_heads, d_ff):
    """
    A single block of the Transformer model. It consists of self-attention, feedforward, and residual connections with layer normalization.

    Parameters:
    - x (numpy.ndarray): Input sequence.
    - d_model (int): Dimensionality of the model.
    - num_heads (int): Number of attention heads.
    - d_ff (int): Dimensionality of the feedforward layer.

    Returns:
    numpy.ndarray: Output after the transformer block.
    """
    # Self-Attention
    self_attention_output = multi_head_attention(x, x, x, num_heads)

    # Residual Connection and Layer Normalization
    x = layer_norm(x + self_attention_output)

    # Feedforward
    feedforward_output = feedforward(x, d_model, d_ff)

    # Residual Connection and Layer Normalization
    x = layer_norm(x + feedforward_output)

    return x

def transformer(input_sequence, d_model, num_heads, d_ff, num_layers):
    """
    Combines multiple transformer blocks to create a complete Transformer model.
    It applies positional encoding to the input sequence before processing it through the transformer blocks..

    Parameters:
    - input_sequence (numpy.ndarray): Input sequence.
    - d_model (int): Dimensionality of the model.
    - num_heads (int): Number of attention heads.
    - d_ff (int): Dimensionality of the feedforward layer.
    - num_layers (int): Number of transformer blocks.

    Returns:
    numpy.ndarray: Output sequence after the transformer model.
    """
    # Positional Encoding
    pos_enc = positional_encoding(len(input_sequence), d_model)

    # Add positional encoding to each position in the input sequence
    x = input_sequence[:, np.newaxis] + pos_enc

    # Transformer Blocks
    for _ in range(num_layers):
        x = transformer_block(x, d_model, num_heads, d_ff)

    return x

if __name__ == "__main__":
    
    # demonstrates the usage of the implemented transformer by applying it to an example input sequence ("Transformer").
    example_input_sequence = "Transformer"

    # Convert input sequence to one-hot encoding
    vocab = {char: idx for idx, char in enumerate(set(example_input_sequence))}
    input_sequence = np.array([vocab[char] for char in example_input_sequence])

    # Transformer Model Parameters
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6

    # Apply the Transformer Model
    output_sequence = transformer(input_sequence, d_model, num_heads, d_ff, num_layers)

    # Display the shape of the output sequence
    print("Output Sequence Shape:", output_sequence.shape)
