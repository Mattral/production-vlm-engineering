import numpy as np

# Define the sequence length and dimensionality
sequence_length = 10
embedding_dim = 8
vocab_size = 20

# Generate random input and target sequences
input_sequence = np.random.randint(0, vocab_size, size=(sequence_length,))
target_sequence = np.random.randint(0, vocab_size, size=(sequence_length,))

# One-hot encode the input and target sequences
def one_hot_encode(sequence, vocab_size):
    """
    One-hot encode a sequence.

    Parameters:
    - sequence (numpy.ndarray): Input sequence.
    - vocab_size (int): Size of the vocabulary.

    Returns:
    numpy.ndarray: One-hot encoded sequence.
    """
    one_hot = np.zeros((len(sequence), vocab_size))
    one_hot[np.arange(len(sequence)), sequence] = 1
    return one_hot

input_one_hot = one_hot_encode(input_sequence, vocab_size)
target_one_hot = one_hot_encode(target_sequence, vocab_size)

# Simple RNN-based Encoder-Decoder
class SimpleEncoderDecoder:
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize a simple RNN-based Encoder-Decoder model.

        Parameters:
        - input_dim (int): Dimensionality of the input sequence.
        - hidden_dim (int): Dimensionality of the hidden states.
        - output_dim (int): Dimensionality of the output sequence.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights and biases for the encoder
        self.W_e = np.random.randn(input_dim, hidden_dim)
        self.U_e = np.random.randn(hidden_dim, hidden_dim)
        self.b_e = np.zeros(hidden_dim)

        # Initialize weights and biases for the decoder
        self.W_d = np.random.randn(hidden_dim, output_dim)
        self.U_d = np.random.randn(output_dim, output_dim)
        self.b_d = np.zeros(output_dim)

    def forward(self, input_sequence):
        """
        Forward pass of the Encoder-Decoder model.

        Parameters:
        - input_sequence (numpy.ndarray): One-hot encoded input sequence.

        Returns:
        list: Decoder outputs at each time step.
        """
        # Encoder forward pass
        encoder_states = []
        h_e = np.zeros(self.hidden_dim)
        for t in range(len(input_sequence)):
            x_t = input_sequence[t]
            h_e = np.tanh(np.dot(x_t, self.W_e) + np.dot(h_e, self.U_e) + self.b_e)
            encoder_states.append(h_e)

        # Decoder forward pass
        decoder_outputs = []
        h_d = np.zeros(self.output_dim)
        for t in range(len(encoder_states)):
            h_d = np.tanh(np.dot(encoder_states[t], self.W_d) + np.dot(h_d, self.U_d) + self.b_d)
            decoder_outputs.append(h_d)

        return decoder_outputs

# Instantiate the model
model = SimpleEncoderDecoder(input_dim=vocab_size, hidden_dim=16, output_dim=vocab_size)

# Forward pass
decoder_outputs = model.forward(input_one_hot)

# Print the input and target sequences, as well as the decoder outputs
print("Input Sequence:")
print(input_sequence)
print("Target Sequence:")
print(target_sequence)
print("Decoder Outputs:")
print(np.argmax(decoder_outputs, axis=1))
