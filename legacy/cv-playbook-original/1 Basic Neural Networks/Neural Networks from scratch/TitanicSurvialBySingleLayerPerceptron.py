import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from SingleLayerPerceptron import SingleLayerPerceptron
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic_data = pd.read_csv("titanic.csv")

# Preprocess the data
# (Assuming you have already loaded and preprocessed the Titanic dataset into X and y)
X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

# Convert categorical variables to numerical
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# Handle missing values (you may need to impute or drop missing values)
X.fillna(0, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Build and train the Single Layer Perceptron
slp = SingleLayerPerceptron(input_size=X_train_std.shape[1])

# Training with print statements and optional plotting
for epoch in range(1, 101):
    slp.train(X_train_std, y_train, epochs=1, learning_rate=0.01)
    predictions = slp.predict(X_test_std)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Epoch: {epoch}, Accuracy: {accuracy}")

    # Plot decision boundary (optional, may be slow for large datasets)
    if epoch % 10 == 0:
        plt.figure(figsize=(8, 6))
        slp.plot_decision_boundary(X_train_std, y_train, epoch, scaler)
        plt.title(f'Decision Boundary - Epoch {epoch}')
        plt.show()
