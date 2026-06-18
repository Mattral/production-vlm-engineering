"""

The Improved Self-Attention Mechanism, commonly used in Transformer models,
has various applications in natural language processing (NLP) and other domains.

Machine Translation:

How it works: Self-attention helps capture relationships between words in different positions in a sentence.
It allows the model to weigh the importance of each word when translating from one language to another.

Example: Translating an English sentence to French, the model can attend to relevant words in the source language
to generate accurate translations.
____________________
Document Classification:

How it works: By capturing dependencies between words in a document,
self-attention improves the understanding of the context and relationships within the text.
Example: Classifying news articles into categories (e.g., sports, politics)
by considering the relationships between words throughout the document.
_________________
Named Entity Recognition (NER):

How it works: Self-attention allows the model to focus on specific parts of a sentence when identifying named entities,
helping in the recognition of entities like names, locations, and organizations.
Example: Extracting names of people, locations, and organizations from a text by attending to relevant parts of the sentence.
__________________
Document Summarization:

How it works: Improved self-attention helps the model identify key information in a document,
facilitating the generation of concise and informative summaries.
Example: Summarizing a long article or document by attending to the most important
information and discarding irrelevant details.
____________________
Question Answering:

How it works: Self-attention aids in understanding the context of a question and
its relation to the surrounding text, enabling accurate identification of answers.
Example: Answering questions about a given passage by attending to relevant sections that contain
the information needed for the answer.
______________________
Sentiment Analysis:

How it works: Self-attention helps the model analyze the sentiment of a text
by considering the relationships between words and capturing nuanced expressions.
Example: Determining the sentiment (positive, negative, neutral) of user reviews by attending to words
that convey emotions and opinion
"""


'''
Let's consider a natural language processing (NLP) use case
where we have a sequence of word embeddings representing a sentence.
We'll use the improved self-attention mechanism to capture contextual information within the sentence.

'''

import numpy as np

class ImprovedSelfAttention:
    def __init__(self, d_model, num_heads):
        """
        Initialize the Improved Self-Attention Mechanism.

        Parameters:
        - d_model (int): Dimensionality of the input embeddings.
        - num_heads (int): Number of attention heads.
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Initialize weight matrices for queries, keys, and values
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        """
        Perform scaled dot-product attention.

        Parameters:
        - Q (numpy.ndarray): Queries.
        - K (numpy.ndarray): Keys.
        - V (numpy.ndarray): Values.

        Returns:
        numpy.ndarray: Attended values.
        """
        scale_factor = np.sqrt(self.head_dim)
        attention_scores = np.matmul(Q, K.T) / scale_factor
        attention_weights = softmax(attention_scores, axis=-1)
        attended_values = np.matmul(attention_weights, V)
        return attended_values



    def split_heads(self, X):
        """
        Split the input into multiple heads.

        Parameters:
        - X (numpy.ndarray): Input.

        Returns:
        List of numpy.ndarray: Input split into heads.
        """
        return np.split(X, self.num_heads, axis=-1)

    def combine_heads(self, X):
        """
        Combine multiple heads into a single tensor.

        Parameters:
        - X (List of numpy.ndarray): Input heads.

        Returns:
        numpy.ndarray: Combined output.
        """
        return np.concatenate(X, axis=-1)

    def self_attention_block(self, X):
        """
        Apply the self-attention block.

        Parameters:
        - X (numpy.ndarray): Input sequence.

        Returns:
        numpy.ndarray: Contextualized output sequence.
        """
        Q = np.dot(X, self.W_q)
        K = np.dot(X, self.W_k)
        V = np.dot(X, self.W_v)

        # Split into heads
        Q_split = self.split_heads(Q)
        K_split = self.split_heads(K)
        V_split = self.split_heads(V)

        # Scaled Dot-Product Attention for each head
        attended_values_split = [self.scaled_dot_product_attention(Q_h, K_h, V_h) for Q_h, K_h, V_h in zip(Q_split, K_split, V_split)]

        # Combine attended values from all heads
        attended_values = self.combine_heads(attended_values_split)

        return attended_values

# Helper function for softmax activation
def softmax(X, axis=-1):
    """
    Apply the softmax activation function.

    Parameters:
    - X (numpy.ndarray): Input.

    Returns:
    numpy.ndarray: Output after applying softmax.
    """
    exp_X = np.exp(X - np.max(X, axis=axis, keepdims=True))
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)

# Other Use Cases:
# 1. Machine Translation: Self-attention helps capture dependencies between words in different languages.
# 2. Document Classification: Capture relationships between words in a document for better classification.
# 3. Named Entity Recognition (NER): Attend to relevant parts of a sentence for identifying named entities.

# Example NLP Use Case
d_model = 256
num_heads = 8
max_sequence_length = 20

# Generate a random sequence of word embeddings for a sentence
word_embeddings = np.random.randn(max_sequence_length, d_model)

# Apply Improved Self-Attention Mechanism
self_attention = ImprovedSelfAttention(d_model, num_heads)
contextualized_sequence = self_attention.self_attention_block(word_embeddings)

# Display the original and contextualized sequences
print("Original Sequence:")
print(word_embeddings)

print("\nContextualized Sequence:")
print(contextualized_sequence)

