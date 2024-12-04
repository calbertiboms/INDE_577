# **Single Neuron Linear Regression Model**

## Overview

This project implements a **Single Neuron Linear Regression Model**, showcasing how a single neuron with a linear activation function can be trained using gradient descent to predict continuous target values. This model effectively demonstrates the fundamentals of **linear regression**, where the neuron computes a weighted sum of input features to make predictions.

In this project, we:
1. Implement a single-neuron model for linear regression.
2. Train the model using gradient descent.
3. Visualize how changes in hyperparameters affect performance.

![Neuron Model](https://raw.githubusercontent.com/RandyRDavila/Data_Science_and_Machine_Learning_Spring_2022/refs/heads/main/Lecture_3/ThePerceptronImage.png)

---
## **How Linear Regression Works**

1. **Model Equation**:
   - Predicts target values (\(y\)) as a weighted sum of input features (\(x\)):
     \[
     y = w \cdot x + b
     \]
2. **Training with Gradient Descent**:
   - Iteratively adjusts weights (\(w\)) and bias (\(b\)) to minimize the cost function.
3. **Cost Function**:
   - Uses Mean Squared Error (MSE) to evaluate prediction accuracy:
     \[
     J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (y_i - (w \cdot x_i + b))^2
     \]
4. **Learning Rate**:
   - Controls the step size in gradient descent, balancing between convergence speed and stability.

---

## **Key Features**

- **Single Neuron with Linear Activation**:
  - The neuron uses a linear activation function to perform regression.
- **Learning Rate Experimentation**:
  - Demonstrates the effects of different learning rates on model convergence and performance.
- **Visualizations**:
  - Subplots illustrate model predictions compared to actual target values and the impact of learning rate adjustments.

---

## **Project Objectives**

1. **Neuron Implementation**:
   - Develop a single neuron to compute predictions for a regression task.
2. **Training Process**:
   - Optimize weights and bias using gradient descent.
3. **Learning Rate Impact**:
   - Train the model with varying learning rates to understand their effect on convergence.
4. **Visualization**:
   - Generate plots to compare predictions and actual values.

---

## **Code Workflow**

### **1. Data Preparation**

- Load or generate a dataset for regression tasks.
- Scale input data for improved gradient descent performance.

### **2. Model Implementation**

- **Neuron Equation**:
  - Implement a neuron that computes a linear weighted sum:
    \[
    y = w \cdot x + b
    \]
- **Gradient Descent**:
  - Adjust weights (\(w\)) and bias (\(b\)) iteratively to minimize error.

### **3. Training**

- Train the neuron on the dataset using gradient descent.
- Explore the impact of different learning rates on model performance.
- Explore the impact of using more features in the analysis.

### **4. Visualization**

- Compare model predictions against actual target values.
- Visualize the effect of learning rate adjustments using subplots.
- Analyze the MSE and MAE reduction. 
- Understand the impact of using more features.

---

## **Results**
- **Prediction Accuracy**:
  - The model successfully fits the data when an appropriate learning rate and amount of features is chosen.
- **Learning Rate Impact**:
  - Smaller learning rates converge slowly but are more stable.
  - Larger learning rates can overshoot the minimum, causing instability.
    
---

## **Conclusion**

This project highlights the simplicity and effectiveness of a single neuron for linear regression. By exploring gradient descent with different learning rates, we gain insight into the trade-offs between convergence speed and stability. The visualizations provide a clear understanding of how well the neuron fits the data, emphasizing the importance of proper hyperparameter tuning. Additionally, by incorporating more features into the model, is possible to observe an improvement in its predictive performance. This demonstrates the importance of feature selection and the impact it has on machine learning models. The model development process is iterative, and experimenting with different approaches is key to achieving the best results.

