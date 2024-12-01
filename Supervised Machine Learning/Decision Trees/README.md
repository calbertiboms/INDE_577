# Decision Trees
## Overview
Decision tree models, like probability tree diagrams, are structured as trees, where each node represents a simple decision point based on a specific feature, leading to branches that guide the model. Starting from the initial node, the model passes through the first decision point and follows a determined branch until reaching another decision point, and follows the same sequential steps until reaching a final point where there are no decisions left.

In this way, the model works like a funel, where possible outcomes are narrowed at each decision level, reaching to the final and unique outcome. 

The goal for this project is to build a Decision Tree Classifier that can say if a person will like or not a song based on previous experiences with different songs. The idea is to simulate how music platforms like Spotify may recommend songs based on similarities with liked songs.

---

## How Decision Trees work
![Decision Tree Diagram](https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/ce/ed/ICLH_Diagram_Batch_03_24A-AI-ML-DecisionTree.png)
This simple case from IBM shows an example on how decisions trees are structured. In this case, we have 3 levels of decisions: the first one asking if there is a swell, the second one asking about the wind level, and the third one asking about the direction. The final outcome will define if it's possible to surf or not. All branches converge at some point to the final decision, but it might take the model to go through different decision levels depending on the path and prior decisions.

Decision trees models can work as classifiers or as regression models. To build a decision tree model, there are also some key hyperparameters to focus on when trying to tune the model for a better performance. The most common ones are:


*   Maximum depth: refers to the depth of the tree
*   Minimum Sample Split: the minimum number of samples required to split an internal node

*   Maximum features: the number of features to consider when looking for the best split
*   Maximum leaf nodes: limits the number of leaf nodes in the tree
---
## Project Objectives

1. **Load Spotify Recommendations Data**: Import a dataset from Kaggle that contains Spotify recommendations based on liked songs.
2. **Preliminaryy Visual Analysis**: Explore possible relationships between songs' features and the target variable.
3. **Apply Decision Tree Classifier**: Build a classifier to predict whether the user will like or not a song based on the selected songs' features.
4. **Visualize Results**: Visualize the Decision Tree algorithm, the decision boundary, the confusion matrix and the Receiver Operating Curve.
---
## Code Workflow
#### 1. Data Loading 
* Load the dataset from Kaggle with zipfile and os.
* Open the dataset with pandas.
#### 2. Preliminary Visual Analysis
* Use visualization libraries.
```python
import matplotlib.pyplot as plt
import seaborn as sns
```
* Build a radar plot with average features.
* Build distribution of features based on liked and not liked songs.
* Build scatterplot for 'danceability' and 'tempo', distinguishing between liked and not liked songs.
#### 3. Decision Tree Classifier 
* Import from sklearn the Decision Tree Classifier, plot tree, classification report and accuracy score.
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
```
* Initialize and fit the model, make predictions and evaluate the model.
#### 4. Visualization
* Visualize decision tree. 
* Visualize tree rules.
* Visualize decision boundary.
* Visualize confusion matrix.
* Visualize Receiver Operating Characteristic Curve.
#### 5. Hyperparameters
* Vary maximum depth to avoid overfitting.
* `max_depth`: longest path from the root node to a leaf node.

## Results
* A Decision Tree Classifier with a maximun depth of 3 was built to predict whether the user would like or not a song based on the song's tempo and durability.
* The model achieved an accuracy of 81%, with a precision score of 92% for liked songs and 78% for not liked songs.
