# Perceptron Project
This project demonstrates the implementation of a **Perceptron**, which is one of the simplest neural networks. 
A perceptron is a binary classifier that attempts to find a hyperplane to separate data into two categories.
It is a single neuron model with the sign activation function as depicted in the figure below.

![Perceptron Model](https://raw.githubusercontent.com/RandyRDavila/Data_Science_and_Machine_Learning_Spring_2022/refs/heads/main/Lecture_3/ThePerceptronImage.png)

### Key Concepts:
- **Perceptron Algorithm**: It updates its weights iteratively based on the input data and the errors produced. The perceptron learns by adjusting these weights to minimize classification errors.
- **Activation Function**: The perceptron uses a step function, where if the weighted sum of the inputs is greater than a threshold, it classifies the input as one category, otherwise as the other.
- **Binary Classification**: The perceptron can only classify linearly separable data, meaning it performs best on datasets where a straight line can divide the categories.

### Highlights:
- This implementation walks through a simple example, adjusting the weights after each iteration to reduce error.
- At the end of the training, you may observe larger errors compared to the start, which could indicate overfitting or limitations in the linear separability of the dataset.

### Requirements:
- Python 3.x
- Libraries: NumPy, Matplotlib (for visualizing the decision boundary)
