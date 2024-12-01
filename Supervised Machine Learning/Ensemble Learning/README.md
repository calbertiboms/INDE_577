# Ensemble Learning
## Overview
Ensemble learning is a machine learning technique that combines predictions from multiple models (also known as 'base learners') to improve the overall performance, accuracy and robustness. The idea is that by aggregating the strengths of different models, the ensemble can outperform any individual model.


The goal of this project is to use ensamble learning to predict employees' satisfaction levels based on different factors. For this purpose, we will use different classifiers from sklearn and then combine them with different ensemble models.

---

## How Ensemble Learning works
There are three main types of ensemble methods:


1.   **Bagging**: trains models on different subsets of the data, reducing variance. An example of bagging can be Random Forest.
2.   **Boosting**: reduces bias by sequentially training models, where each model focuses on correcting the errors of its predecessor. An example can be Boosting.
3. **Stacking**: it uses a 'meta model' to learn the best way to blend predictions from multiple models. An example is using logistic regression to combine oredictions from decision trees and SVMs.

In this way, bagging works with parallel models, boosting works with sequential models, and stacking combines predictions from multiple models by using a meta-model to learn the best way to integrate their outputs

The structure of these methods can be seen in the picture below, taken from [Spot Intelligence](https://spotintelligence.com/2023/08/09/ensemble-learning/)

![Bagging, Boosting, and Stacking](https://i0.wp.com/spotintelligence.com/wp-content/uploads/2024/03/bagging-boosting-stacking-1024x576.webp?resize=1024%2C576&ssl=1)

---
## Project Objectives

1. **Load Employee Satisfaction Survey Data**: Import a dataset from Kaggle that contains the survey.
2. **Preliminary Visual Analysis**: Explore possible relationships between variables to determine features for the model.
3. **Build Individual Models**: Construct indiviudal models (Logistic Regression, Random Forest Classifier and SVC) to predict if the employee is satisfied or not and calculate their accuracy.
4. **Build Ensemble Model: Voting Classifier**: Build the Voting Classifier combining the individual models and calculate its accuracy.
5. **Build the Ensemble Model: Decision Trees Classifier**: Build the Decision Trees Classifier combining the individual models and calculate its accuracy.
---
## Code Workflow
#### 1. Data Loading and Preprocessing
* Load the dataset from Kaggle with zipfile and os.
* Open the dataset with pandas.
* To determine the satisfaction levels, we will consider that employees with satisfaction_level > 0.5 are 'satisfied' (1), and others as 'not satisfied' (0)
```python
data['satisfied'] = (data['satisfaction_level'] > 0.5).astype(int)
```
* One-hot encode categorical variables 
```python
X = pd.get_dummies(X, drop_first=True)
```
#### 2. Preliminary Visual Analysis
* Use visualization libraries.
```python
import matplotlib.pyplot as plt
import seaborn as sns
```
* Build distribution of satisfaction level to see how the sample distribution is.
* Build density plot to study satisfaction level according to last evaluation.
* Build a boxplot for satisfaction level vs. number of projects.
* Build a boxplot to study satisfaction level vs. department.
* Build a boxplot to analyze satisfaction level based on salary level.

#### 3. Individual Models
* Import from sklearn the Logistic Regression, Random Forest and SVC classifiers.
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
```
* Initialize and fit the models, make predictions and evaluate the models with their respective accuracy scores.
#### 4. Ensemble Models 
* Import ensemble models (Voting Classifier, Decision Trees Classifier) from sklearn
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
```
* Define estimators. 
* Build models, make predictions and build classification report.
#### 5. Hyperparameters for Decision Trees
* Vary maximum depth and study differences in accuracy and performance.
* `max_depth`: longest path from the root node to a leaf node..

## Results
* We built three individual models to predict an employee's satisfaction based on several features. The results showed that the Random Forest Classifier was the one with the highest accuracy.

| Classifier               | Accuracy |
|--------------------------|----------|
| Support Vector Machine   | 0.70     |
| Logistic Regression      | 0.68     |
| Random Forest            | 0.87     |

* We built ensemble learning models and obtained the following results.

| Ensemble Model                  | Accuracy |
|------------------------|----------|
| Stump Classifier       | 0.77     |
| Tree Classifier        | 0.85     |
| Voting Classifier      | 0.73     |

* In conclusion, although the ensemble models performed better than the Logistic Regression and SVC individual models, the Random Forest obtained the highest accuracy. 
