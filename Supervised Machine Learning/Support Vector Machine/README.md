# **Support Vector Machine**

## Overview

Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification and regression tasks. It is particularly effective for problems with clear margins of separation between classes. SVM works by finding the optimal hyperplane that maximizes the margin between different classes, making it a robust choice for both linear and non-linear problems.

![Support Vector Machine](https://miro.medium.com/max/1400/1*KMi_f-m3O10-3K8N1w2bDA.png)

This notebook explores SVM to predict class labels for a dataset using different kernel functions. SVM is implemented using `scikit-learn`, and the model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

In this project, we:
1. Preprocess the dataset for machine learning.
2. Train the SVM model with various kernels (linear, RBF, polynomial).
3. Evaluate and compare the performance of different configurations.
4. Visualize model metrics with confusion matrices and ROC curves.

---

## **How Support Vector Machine Works**
1. **Model Equation**: Finds the hyperplane that separates classes by maximizing the margin.
2. **Training**: Uses optimization to determine support vectors that define the hyperplane.
3. **Prediction**: Classifies inputs based on their position relative to the hyperplane.

**Key Features**:
- Effective for binary and multi-class classification.
- Robust to high-dimensional data.
- Flexible through kernel tricks to handle non-linear relationships.

---

## **Project Objectives**
1. **Dataset Preparation**:
   - Handle missing values, scale features, and split the data into training and testing sets.
2. **Model Implementation**:
   - Train SVM models with different kernels (`linear`, `RBF`, `polynomial`, `sigmoid`).
   - Tune hyperparameters such as `C` and `gamma` to optimize performance.
3. **Performance Evaluation**:
   - Assess performance using classification reports, confusion matrices, and ROC-AUC scores.

---

## **Code Workflow**

### **1. Data Preprocessing**
- Import the dataset and clean missing values.
- Scale features using `StandardScaler`.
- Split data into training and testing sets using an 80/20 split.

### **2. SVM Implementation**
- Train SVM models with different kernels (`linear`, `RBF`, `polynomial`, `sigmoid`).

### **3. Evaluation**
- Evaluate model performance using metrics like:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Visualize results with confusion matrices and ROC curves.

---

## **Results**

| Kernel      | C       | Accuracy |
|-------------|---------|----------|
| Default     | Default | 0.9827   |
| RBF         | 100     | 0.9832   |
| RBF         | 1000    | 0.9816   |
| Linear      | 1       | 0.9816   |
| Linear      | 100     | 0.9832   |
| Polynomial  | 1       | 0.9807   |
| Polynomial  | 100     | 0.9824   |
| Sigmoid     | 1       | 0.8858   |
| Sigmoid     | 100     | 0.8855   |

---

## **Conclusion**

### Best Model
The linear kernel with `C=100.0` was one of the best-performing model based on the results. It achieved the highest overall accuracy (98.32%) and demonstrated strong precision (0.93) and recall (0.84) for the minority class. These metrics indicate that the linear kernel effectively separates the classes while minimizing misclassification.


---