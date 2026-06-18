import numpy as np
import matplotlib.pyplot as plt


def positional_encoding(max_len, d_model):
    """
    Generate positional encoding for a sequence.

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

if __name__ == "__main__":
    max_len = 50  # Maximum length of the sequence
    d_model = 512  # Dimensionality of the model

    pos_encoding = positional_encoding(max_len, d_model)

    # Display the positional encoding for the first 10 positions and first 10 dimensions
    print("Positional Encoding:")
    print(pos_encoding[:10, :10])

# Function to visualize embeddings
def plot_embeddings(embeddings, title):
    plt.figure(figsize=(12, 6))
    plt.imshow(embeddings, cmap='viridis', aspect='auto')
    plt.colorbar(label='Embedding Value')
    plt.title(title)
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.show()

# Generate word embeddings for a sequence (for illustration purposes)
sequence_length = 20
embedding_dim = 512
word_embeddings = np.random.randn(sequence_length, embedding_dim)

# Generate positional encodings
pos_encodings = positional_encoding(sequence_length, embedding_dim)

# Add positional encodings to word embeddings
embedded_sequence = word_embeddings + pos_encodings

# Visualize the original word embeddings
plot_embeddings(word_embeddings.T, title='Original Word Embeddings')

# Visualize the embeddings after adding positional encodings
plot_embeddings(embedded_sequence.T, title='Word Embeddings with Positional Encodings')
