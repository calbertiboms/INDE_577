# **Perceptron**

## Overview 
The Perceptron is one of the simplest and earliest artificial neural networks used for binary classification. It classifies data by learning a linear decision boundary through an iterative training process. This project demonstrates the working mechanism of the Perceptron algorithm and explores how different learning rates influence its performance.

![Perceptron Model](https://raw.githubusercontent.com/RandyRDavila/Data_Science_and_Machine_Learning_Spring_2022/refs/heads/main/Lecture_3/ThePerceptronImage.png)

In this project, we:
1. Generate and visualize a synthetic dataset for binary classification.
2. Implement the Perceptron algorithm in Python.
3. Train the model and visualize its learning progress.
4. Compare the effects of varying learning rates on its performance.

---

## **How the Perceptron Works**
1. **Initialization**:
   - The Perceptron starts with random weights and a bias.
2. **Training Process**:
   - For each data point, it computes the weighted sum and applies an activation function to predict the class.
   - If the prediction is incorrect, it updates the weights and bias based on the error.
3. **Decision Boundary**:
   - Over multiple iterations, the Perceptron learns a linear boundary that separates the two classes.

**Key Features**:
- Simple to implement and computationally efficient.
- Effective for linearly separable data.
- Iterative weight update mechanism ensures continuous learning.

---

## **Project Objectives**
1. **Generate Synthetic Data**:
   - Create a dataset with two features and two classes for binary classification.
2. **Implement Perceptron Algorithm**:
   - Develop a `Perceptron` class with methods for training and prediction.
3. **Visualize Learning Progress**:
   - Plot the decision boundary after each training iteration.
4. **Analyze Learning Rates**:
   - Train the Perceptron with different learning rates and observe their impact on convergence and accuracy.

---

## **Code Workflow**

### **1. Data Generation and Visualization**
- Generate a 2D synthetic dataset with two features and two classes.
- Visualize the dataset using a scatter plot.

### **2. Implementing the Perceptron**
- Create a `Perceptron` class with methods for training and prediction.
- The training process iteratively updates weights and bias to minimize classification errors.

### **3. Training and Visualizing Decision Boundary**
- Train the Perceptron on the dataset.
- Visualize the decision boundary to observe the classification.

### **4. Comparing Learning Rates**
- Train with different learning rates (e.g., 0.01, 0.1).
- Compare the decision boundaries and performance.

---
## **Results**
- Decision boundary: the perceptron successfully learned a linear boundary to separate the two classes.
- Effect of learning rates: the selection of the learning rate can change the outcome. 

---
## **Conclusion**
The Perceptron algorithm is a foundational concept in machine learning, showing how a simple iterative process can achieve effective classification for linearly separable data. This notebook provides an interactive way to explore its workings and understand the importance of hyperparameters such as the learning rate.

