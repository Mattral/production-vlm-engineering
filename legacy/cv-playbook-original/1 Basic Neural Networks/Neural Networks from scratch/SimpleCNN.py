import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        Initialize a 2D convolutional layer.

        Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (int): Size of the convolutional kernel.
        """
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.zeros((out_channels, 1))

    def forward(self, x):
        """
        Perform forward pass for the convolutional layer.

        Parameters:
        - x (numpy array): Input data.

        Returns:
        - numpy array: Output data after convolution.
        """
        self.last_input = x
        batch_size, in_channels, in_height, in_width = x.shape
        out_channels, _, kernel_size, _ = self.weights.shape
        out_height = in_height - kernel_size + 1
        out_width = in_width - kernel_size + 1
        output = np.zeros((batch_size, out_channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                output[:, :, i, j] = np.sum(
                    x[:, np.newaxis, :, i:i+kernel_size, j:j+kernel_size] * self.weights,
                    axis=(2, 3, 4)  # Adjust axis for correct summation
                )

        output += self.bias.reshape(1, -1, 1, 1)
        return output

    def plot_filters(self):
        """
        Plot the filters of the convolutional layer.
        """
        out_channels, in_channels, _, _ = self.weights.shape

        # Handle the case when either out_channels or in_channels is 1
        if out_channels == 1 or in_channels == 1:
            fig, axes = plt.subplots(max(out_channels, in_channels), figsize=(in_channels, out_channels))
        else:
            fig, axes = plt.subplots(out_channels, in_channels, figsize=(in_channels, out_channels))

        # Iterate over the flattened axes array
        for i, ax in enumerate(axes.flatten()):
            row_index = i // in_channels
            col_index = i % in_channels

            # Plot flattened weights
            ax.plot(self.weights[row_index, col_index].reshape(-1), color='black')
            ax.axis('off')

        plt.show()

class MaxPool2D:
    def __init__(self, pool_size):
        """
        Initialize a 2D max pooling layer.

        Parameters:
        - pool_size (int): Size of the pooling window.
        """
        self.pool_size = pool_size

    def forward(self, x):
        """
        Perform forward pass for the max pooling layer.

        Parameters:
        - x (numpy array): Input data.

        Returns:
        - numpy array: Output data after max pooling.
        """
        self.last_input = x
        batch_size, in_channels, in_height, in_width = x.shape
        pool_height, pool_width = self.pool_size, self.pool_size
        out_height = in_height // pool_height
        out_width = in_width // pool_width
        output = np.zeros((batch_size, in_channels, out_height, out_width))

        for i in range(out_height):
            for j in range(out_width):
                output[:, :, i, j] = np.max(
                    x[:, :, i*pool_height:(i+1)*pool_height, j*pool_width:(j+1)*pool_width],
                    axis=(2, 3)
                )

        return output

class Flatten:
    def forward(self, x):
        """
        Perform forward pass for flattening layer.

        Parameters:
        - x (numpy array): Input data.

        Returns:
        - numpy array: Flattened output data.
        """
        self.last_input_shape = x.shape
        return x.reshape(x.shape[0], -1)

class Dense:
    def __init__(self, in_features, out_features):
        """
        Initialize a fully connected layer.

        Parameters:
        - in_features (int): Number of input features.
        - out_features (int): Number of output features.
        """
        self.weights = np.random.randn(out_features, in_features)
        self.bias = np.zeros((out_features, 1))

    def forward(self, x):
        """
        Perform forward pass for the fully connected layer.

        Parameters:
        - x (numpy array): Input data.

        Returns:
        - numpy array: Output data after fully connected layer.
        """
        self.last_input = x
        return np.dot(self.weights, x.T).T + self.bias.T

class SimpleCNN:
    def __init__(self, in_channels, num_classes):
        """
        Initialize a simple CNN model.

        Parameters:
        - in_channels (int): Number of input channels.
        - num_classes (int): Number of output classes.
        """
        self.conv1 = Conv2D(in_channels, 32, kernel_size=3)
        self.pool1 = MaxPool2D(2)
        self.flatten = Flatten()
        self.dense1 = Dense(5408, 128)
        self.dense2 = Dense(128, num_classes)

    def forward(self, x):
        """
        Perform forward pass for the entire CNN model.

        Parameters:
        - x (numpy array): Input data.

        Returns:
        - numpy array: Output data after passing through the model.
        """
        x = self.conv1.forward(x)
        x = self.pool1.forward(x)
        x = self.flatten.forward(x)
        x = self.dense1.forward(x)
        x = self.dense2.forward(x)
        return x

"""
____________Syntatic DATA__________
"""
# Example usage with random input
model = SimpleCNN(in_channels=1, num_classes=10)
input_data = np.random.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
output = model.forward(input_data)
print(output)

# Plot the learned filters
model.conv1.plot_filters()

'''
___________________Image DATA____________
'''
# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((28, 28))  # Resize the image to match the input size of the CNN
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.transpose(image, (0, 3, 1, 2))  # Adjust the channels axis if needed
    return image

# Example usage with image loading
model = SimpleCNN(in_channels=3, num_classes=10)  # Assuming 3 channels for RGB images
image_path = "kali.jpg"
input_image = load_and_preprocess_image(image_path)
output = model.forward(input_image)
print(output)

# Plot the learned filters
model.conv1.plot_filters()
