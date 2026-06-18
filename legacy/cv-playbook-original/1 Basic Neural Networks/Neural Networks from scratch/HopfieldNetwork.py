'''
 A Hopfield Network is a type of recurrent artificial neural network
 that serves as a content-addressable memory system.
 It is used for pattern recognition, particularly in the realm of associative memory.
 Hopfield Networks have applications in various fields,
 such as image recognition, optimization problems, and information retrieval.

Please note that Hopfield Networks are typically used for
associative memory and pattern recall
rather than classification, regression, computer vision, or NLP.
'''


"""
Explanation:

-The HopfieldNetwork class is initialized with a size representing the number of neurons in the network.
-The train method is used to train the network on a set of patterns.
-The recall method is used to recall a pattern given an input pattern.
-The visualize_patterns function is used to visualize the original and reconstructed patterns.
"""

import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, size):
        """
        Initialize a Hopfield Network with a given size.

        Parameters:
        - size (int): Number of neurons in the network.
        """
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """
        Train the Hopfield Network on a set of patterns.

        Parameters:
        - patterns (numpy.ndarray): Array of patterns to train the network.

        Returns:
        None
        """
        for pattern in patterns:
            pattern = pattern.reshape((-1, 1))
            self.weights += np.outer(pattern, pattern)
            np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, max_iter=100):
        """
        Recall a pattern from the Hopfield Network.

        Parameters:
        - pattern (numpy.ndarray): Input pattern to recall.
        - max_iter (int): Maximum number of iterations for pattern recall.

        Returns:
        numpy.ndarray: Recalled pattern.
        """
        pattern = pattern.reshape((-1, 1))
        for _ in range(max_iter):
            activation = np.dot(self.weights, pattern)
            pattern = np.sign(activation)
        return pattern

def visualize_patterns(patterns, reconstructed_patterns, title):
    """
    Visualize original and reconstructed patterns side by side.

    Parameters:
    - patterns (numpy.ndarray): Array of original patterns.
    - reconstructed_patterns (numpy.ndarray): Array of reconstructed patterns.
    - title (str): Title for the visualization.

    Returns:
    None
    """
    num_patterns = len(patterns)

    plt.figure(figsize=(10, 4 * num_patterns))
    for i in range(num_patterns):
        plt.subplot(num_patterns, 2, 2 * i + 1)
        plt.imshow(patterns[i].reshape((int(np.sqrt(len(patterns[i]))), -1)), cmap='gray')
        plt.title(f'Original Pattern {i + 1}')

        plt.subplot(num_patterns, 2, 2 * i + 2)
        plt.imshow(reconstructed_patterns[i].reshape((int(np.sqrt(len(reconstructed_patterns[i]))), -1)), cmap='gray')
        plt.title(f'Reconstructed Pattern {i + 1}')

    plt.suptitle(title)
    plt.show()

# Example Usage
pattern_size = 25
patterns = np.array([[1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1],
                     [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                     [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1]])

hopfield_net = HopfieldNetwork(pattern_size)
hopfield_net.train(patterns)

# Test recall with noisy patterns
noisy_patterns = np.array([[1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1],
                           [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
                           [1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]])

reconstructed_patterns = [hopfield_net.recall(pattern) for pattern in noisy_patterns]

# Visualize the patterns
visualize_patterns(noisy_patterns, reconstructed_patterns, "Hopfield Network - Original vs Reconstructed Patterns")

