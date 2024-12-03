# **Logistic Regression**

## Overview

Logistic regression is a statistical model used for binary classification tasks, where the target variable has two possible outcomes (e.g., "on time" or "delayed"). It works by modeling the relationship between the features and the probability of an event occurring using a sigmoid function, which maps inputs to a value between 0 and 1. Logistic regression is widely used for its simplicity, interpretability, and effectiveness in many real-world scenarios.

![Logistic Regression](https://miro.medium.com/v2/resize:fit:1280/1*OUOB_YF41M-O4GgZH_F2rw.png)

This notebook explores logistic regression to predict shipment delays using a supply chain logistics dataset. Logistic regression is a supervised learning algorithm used for binary classification problems, offering interpretable results and robust performance.

In this project, we:
1. Preprocess the dataset for machine learning.
2. Implement logistic regression using `scikit-learn` and a custom implementation.
3. Evaluate and interpret model performance.
4. Compare feature selection approaches and model evaluation techniques.

---

## **How Logistic Regression Works**
1. **Model Equation**: Uses a sigmoid function to model the probability of a binary outcome.
2. **Training**: Optimizes weights to minimize the error between predicted probabilities and actual labels.
3. **Prediction**: Classifies inputs based on the probability threshold (default: 0.5).

**Key Features**:
- Effective for binary classification.
- Provides interpretable coefficients for feature impact.
- Handles linearly separable data well.

---

## **Project Objectives**
1. **Dataset Preparation**:
   - Handle missing values, scale features, and balance the classes using downsampling, SMOTE and downsampling+PCA and SMOTE+PCA.
2. **Model Implementation**:
   - Develop a custom logistic regression algorithm for deeper understanding.
   - Use `scikit-learn` for train test split, classification report, confusion matrix, PCA and SMOTE. 
3. **Performance Evaluation**:
   - Classification report and confusion matrix to understand any classification problems due to the unbalanced dataset.

---

## **Code Workflow**

### **1. Data Preprocessing**
- Import the dataset and clean missing values.
- Balance the dataset using downsampling and SMOTE. Used PCA to improve the model. 
- Split data into training and testing sets.

### **2. Logistic Regression Implementation**
- Build a custom implementation to learn the internal mechanics.

### **3. Evaluation**
- Evaluate model performance using metrics like:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Visualize results with the confusion matrix and classification report. 

---

## **Results**

| Method               | Overall Accuracy | Delayed Shipments Precision | Delayed Shipments Recall |
|-----------------------|------------------|-----------------------------|---------------------------|
| Downsampling          | 0.69            | 0.10                        | 0.10                      |
| SMOTE                 | 0.59            | 0.59                        | 0.58                      |
| Downsampling + PCA    | 0.74            | 0.24                        | 0.21                      |
| SMOTE + PCA           | 0.55            | 0.55                        | 0.55                      |

---

---

## **Conclusion**

### Best Model
The downsampling + PCA model is the best-performing model based on these results. It achieves the highest overall accuracy (74%) and demonstrates an improvement in recall (0.21) and precision (0.24) for delayed shipments compared to other models. The SMOTE models underperformed because the synthetic oversampling caused the predicted odds for not delayed shipments to be closer to 50%, which is standard for random guessing, reducing the effectiveness.

This suggests that PCA effectively retains key information while reducing dimensionality, allowing the model to better generalize and capture patterns in delayed shipments. Compared to the purely downsampled or SMOTE models, the downsampling + PCA model provides a more balanced approach, enhancing the model’s ability to identify minority class instances without sacrificing overall accuracy.

