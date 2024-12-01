# Principal Components Analysis
## Overview

PCA is an unsupervised machine learning model that intends to reduce the dimensionality of the data while preserving as much variability as possible.

It transforms the original features (given by the columns) into a smaller set of new uncorrelated variables called principal components. These components are linear combinations of the original features.

PCA is commonly use to simplify datasets, reduce noise and avoid collinearilty. It's often used as an initial data process before some other supervised learning technique to predict a target variable.

For this project, we will study a dataset of around 27000 images of women and men and we will intend to recognize the gender of the person in the image using PCA and a supervised learning classifier.

---

## How PCA works
PCA finds new axes (principal components) that capture the maximum variance in the data, allowing for simpler visualization or computation without losing significant information.

For this purpose, the algorithm goes through these five steps: 

1. **Standardization**: The data is standardized to have a mean of 0 and unit variance.
2. **Covariance Matrix**: A covariance matrix is calculated to measure the relationships between features.
3. **Eigenvalues and Eigenvectors**: The eigenvalues (variance explained) and eigenvectors (directions of variance) of the covariance matrix are computed.
4. **Principal Components**: Eigenvectors are ranked by their corresponding eigenvalues, and the top components are chosen.
5. **Projection**: The data is projected onto these principal components to reduce dimensionality while preserving the most important patterns.

The following image from [Medium](https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d) shows an example of how PCA can help to reduce the dimensionality of a 3D dataset by seeing the projection in a 2D plane.

![PCA Visualization](https://miro.medium.com/v2/resize:fit:972/1*Pogui8xj5BWn5Yc6CFMv8g.png)





---
## Project Objectives

1. **Load Women/Men Photos Dataset**: Load 27000 images from Kaggle.
2. **Oversample**: Oversample women's photos to avoid errors related to imbalances in the dataset.
3. **Perform Principal Component Analysis with 20 components**: Reduce the dimensionality of the photos from 10,000 to 20 components.
4. **Build Logistic Regression Classifier**: Build a Logistic Regression classifier to predict from the 20 components of the image whether the person is a man or a woman. Run the model and study performance with the classification report and the confusion matrix.
5. **Perform Elbow Method to determine optimal number of components for PCA**: Plot the cumulative variance ratio against the number of components to identify the 'elbow point' where the rate of explained variance starts to level off.
6. **Perform Principal Component Analysis with optimal number of components**: Reduce the dimensionality of the photos from 10,000 to the number of components selected with the Elbow Method Analysis.
7. **Run Logistic Regression Classifier again**: Build the classifier, fit it to the dataset reduced with PCA, make predictions and evaluate the model with the classification report and confusion matrix.
8. **Reconstruct images**: Visualize how the PCA is reconstructing the images.
## Code Workflow
#### 1. Data Loading and Preprocessing
The dataset consists of grayscale images of two categories: "man" and "woman." The preprocessing steps are performed as follows:

1. **Directory Structure**:
   - The data is stored in subfolders named `man` and `woman` within the main folder (`base_dir`).
   - Each subfolder contains images for its respective category.

2. **Image Loading**:
   - Images are loaded from their respective folders using OpenCV (`cv2.imread`).
   - Only valid images are processed, and invalid files are ignored.

3. **Grayscale Conversion**:
   - All images are read in grayscale mode to simplify the data and reduce dimensionality.

4. **Resizing**:
   - Each image is resized to a standard size of 100x100 pixels to ensure uniformity across the dataset.

5. **Flattening**:
   - The resized images are flattened into 1D arrays to make them compatible with machine learning models.

6. **Label Assignment**:
   - Images in the `man` folder are assigned the label `0`.
   - Images in the `woman` folder are assigned the label `1`.

7. **Output**:
   - The processed images are stored in the `X` array.
   - The corresponding labels are stored in the `y` array.

Example Usage:
The preprocessing function is used to load and process the dataset:
```python
X, y = preprocess_images(base_dir)
print(f"Loaded {len(X)} images.")
```
#### 2. Oversampling
To correct imbalances in the dataset, we oversample women's photos to match the number of men's photos using 'resample' from scikit-learn.
```python
from sklearn.utils import resample
X_female_oversampled = resample(
    X_female,
    replace=True,                 
    n_samples=len(X_male),          
    random_state=42
```
#### 3. PCA with 20 components
* Apply PCA (from scikit-learn) with 20 components to the dataset.
```python
from sklearn.decomposition import PCA
```
#### 4. Logistic Regression Classifier (20 features)
* Import the Logistic Regression Classifier from scikit-learn.
```python
from sklearn.linear_model import LogisticRegression
```
* Build the model, fit it to the dataset that has been reduced with PCA, make predictions and evaluate the model with a classification report and a confusion matrix.
```python
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```
#### 5. Elbow Method
* Plot the cumulative variance ratio against the number of components (from 1 to 500) to find the 'elbow point'.
* Visually define the optimal number of components for PCA.

#### 6. PCA with 150 components
* Apply PCA (from scikit-learn) with 150 components to the dataset.

#### 7. Logistic Regression Classifier (150 features)
* Build the model, fit it to the dataset that has been reduced with PCA, make predictions and evaluate the model with a classification report and a confusion matrix.
#### 8. Visualization
* Reconstruct the images.
```python
X_train_reconstructed = pca.inverse_transform(X_train_pca)
```

#### 9. Self predictions
* Use the model to predict our genders (Cecilia and Valentina).
## Results
* With PCA, images with 10,000 features (given by 100x100 pixels) were reduced to 150 features (98.5% dimensionality reduction). 
* This dimensionality reduction allowed a supervised classifier to quickly learn from the training set and make predictions that achieved a 72.2% accuracy, with a recall of 73% for men and 71% for women. These are really good results considering the 98.5% reduction in explanatory features.
* PCA was a critical step in building the model. Without it, the Logistic Regression classifier would have struggled to process the high-dimensional data due to computational and memory constraints.
