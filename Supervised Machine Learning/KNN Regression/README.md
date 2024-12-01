# K-Nearest Neighbors
## Overview
K-Nearest Neighbors is a simple supervised learning model that works based on the principle of similarity, assuming that similar data points are likely to have similar outcomes. This algorithm can be used both for classification or regression.

KNN is non-parametric, which means that it does not make assumptions on how the data is distributed. For this reason, this model is very useful when the relationships within the data are complex and difficult to model mathematically.

For this project, we will implement the KNN regression model to predict the critical temperature for an unknown superconductor material based on other characteristics.

The main reason for applying KNN for this problem is because superconductors with similar atomic or physical characteristics tend to have similar critical temperatures. KNN models start from the assumption that groups have similar behaviors, which is why it can help to predict the critical temperature by assuming that this value will be similar to that of other materials with similar features. Moreover, KNN does not make strong assumptions about the distribution of the data - it's flexible and useful in cases where there isn't a clear mathematical model to relate the features.

In this way, for each new material whose critical temperature we want to predict, the algorithm will pick the average of the k nearest neighbors in the dataset.

---
## How K-Nearest Neighbors works
For a determined data point, the model identifies its k nearest neighbors in the feature space using a distance metric (in general, the Euclidean distance). For regression, it predicts the output based on the average of the target value of these neighbors; while in classification models, it assigns the predominant class of the neighbors.

![KNN](https://miro.medium.com/v2/resize:fit:505/0*2_qzcm2gSe9l67aI.png)
The KNN algorithm often works with one of these distance metrics.
Euclidean distance: 

$$
\text{distance}(p, q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}
$$


Manhattan distance:

$$
\text{distance}(p, q) = \sum_{i=1}^n |p_i - q_i|
$$

Minkowski distance:

$$
\text{distance}(p, q) = \left( \sum_{i=1}^n |p_i - q_i|^p \right)^{\frac{1}{p}}
$$


---
## Project Objectives

1. **Load Superconductors Data**: Import a dataset from Kaggle with superconductors features.
2. **Preliminary Visual Analysis**: Explore if there is a possible group pattern for datapoints to justify the use of KNN.
3. **PCA for dimensionality reduction**: Perform the unsupervised learning PCA to reduce the number of features for the model. 
4. **Build KNN Regression model using sklearn**: Build a KNN regression model to predict the critical temperature for an unkown superconductor using sklearn.
5. **Find the optimal k for the model**: Look for the optimal number of neighbors to minimize the MSE and maximize the accuracy of the model.
6. **Build a KNN Regression model from scratch**: Build the necessary functions to perform the KNN algorithm without using libraries.
7. **Try different distance metrics with the model built from scratch**: Try with Manhattan distance and Minkowski distance and study impact on overall performance.
8. **Compare models:** Compare results from built-from-scratch models with different distance metrics and the model from sklearn.
---
## Code Workflow
#### 1. Data Loading 
* Load the dataset from Kaggle with zipfile and os.
* Open the dataset with pandas.
* Standardize features.
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
#### 2. Preliminary Visual Analysis
* Use visualization libraries.
```python
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
```
* Build the distribution of critical temperatures. 
* Divide datapoints in two groups: those with high critical temperature and those with low critical temperature and try to find group patterns. 
* Graph group behaviour in terms of fusion heat, weighted entropy and mean atomic mass.
* Graph group behaviour in terms of mean electron affinity, weighted mean thermal conductivity and mean valence.
#### 3. Principal Components Analysis (PCA)
* Use PCA to reduce the dimensionality of the 82 features.
```python
from sklearn.decomposition import PCA
```
* Plot explained variance by PCA components to determine an adequate number of components.
* Define 15 features for the model and reduce the dimension of the dataset. 
#### 4. KNN Regression model from scikit-learn
* Build KNN Regression model. 
* Evaluate the model with MSE, MAE and r2. 
```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```
#### 6. Find optimal *k* neighbors 
* Test k from 1 to 20 and build their models.
* Make predictions.
* Plot MSE and r2. 
* Choose k=4 to maximize accuracy, minimize MSE and avoid overfitting.
#### 7. KNN Regression model from scratch
* Define Euclidean distance function.
* Define a function to calculate k-nearest neighbors.
* Define a function to make predicitions.
* Define a function to calculate the MSE.
* Define a function to calculate the r2.
* Build the model with k=4, fit it to the data and analyze performance.
#### 8. KNN Regression model from scratch with other distance metrics
* Define Manhattan distance function.
* Define Minkowski distance function.
* Modify function to make predicitions to accept a distance function.   
* Make predictions with both distance metrics and study performance of the models.
---
## Results
* We used PCA to reduce the dimensionality of the dataset from 82 features to 15 features. 
* We built four KNN Regression models: one with sklearn, and three from scratch using different distance metrics.

| Model                    | Mean Squared Error (MSE) | Mean Absolute Error (MAE) | R-squared (R²) |
|--------------------------|--------------------------|----------------------------|----------------|
| KNN Manhattan Distance   | 126.49                  | -                          | 0.8910         |
| KNN Minkowski Distance    | 129.60                  | -                          | 0.8883         |
| KNN Euclidean Distance    | 124.87                  | -                          | 0.8924         |
| KNN (scikit-learn)        | 135.18                  | 6.50                       | 0.8835         |
* We noticed that the best performing model was the one that we built from scratch using the Euclidean distance as the distance metric.
