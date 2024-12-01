# K-Means Clustering 
## Overview

Unsupervised learning is a type of machine learning where the model is trained on data without labeled responses. In this way, the goal of some unsupervised learning algorithms, like K-Means Clustering, is to discover hidden patterns in the data and find the labels.

Unsupervised learning models intend to identify groupings in the data with no predefined labels or outcomes. Particularly, K-Means Clustering is a clustering algorithm used to classify a dataset into *k* distinct, non-overlapping groups or clusters based on similarity.

The idea of this project is to use K-Means clustering to find similar counties in the US according to common economic demographics. For this purpose, we will use data from the US census for each county.


---

## How K-Means Clustering works
The algorithm works as follows:



1.   The model is initialized by selecting *k* random points as initial cluster centroids.
2.   Each data point is assigned to a cluster. This is done by calculating the distances between each point and all the centroids and selecting the closest one. There are different ways of calculating the distance, but the most common method is the Euclidean distance.
3. Once every data point is assigned to a cluster, the algorithm proceeds to update the centroids by taking the mean of all the points in that cluster.
4. The algorithm re-starts with the new centroids and defines new clusters until the centroids stabilize or a specific number of iterations is reached.

The key parameters for this model are:

* **Number of clusters (k)**: number of groups to divide the dataset.
* **Distance metric:** usually Euclidean distance.

$$
\text{distance}(p, q) = \sqrt{\sum (p_i - q_i)^2}
$$
 ![K-Means Clustering Algorithm](https://static.javatpoint.com/tutorial/machine-learning/images/k-means-clustering-algorithm-in-machine-learning.png)

**Defining *k*, the number of clusters**

The **elbow method** is a technique used to determine the optimal number of clusters in K-Mean Clustering. It works by plotting the inertia (sum of squared distances between data points and their assigned clsuter centroids) for different values of k.

The idea behind this methodology is that initially, as k increases, the inertia dicreases significantly because more clusters allow points to be closer to their centroids. However, after a certain point, also known as the 'elbow', the decrease in inertia becomes marginal. At this point, adding more clusters provides diminishing returns since the model strats to overfit the data.

This method balances compact clusters and simplicity.

**Inertia Calculation**

The inertia is calculated as the sum of squared distances between each data point and its closest cluster centroid:

$$
\text{Inertia} = \sum_{i=1}^{n} \min_{k} \|x_i - c_k\|^2
$$

Where:
- \( n \) is the total number of data points,
- \( x_i \) is a data point,
- \( c_k \) is the centroid of cluster \( k \),
- \( \|x_i - c_k\|^2 \) is the squared distance between the point \( x_i \) and the centroid \( c_k \),
- \( \min_{k} \) finds the closest centroid to \( x_i \).

---
## Project Objectives

1. **Load US Census Data**: Import last US Census data from Kaggle.
2. **Preliminary Visual Analysis**: Study some map graphs showing povery, unemployment and GDP per capita by county; and graph a 3D scatterplot to visualize possible groups according to these three features.
3. **Build K-Means Clustering model from scratch**: Define functions to do K-Means Clustering algorithm and identify clusters.
4. **Define *k* with the Elbow Method**: Plot the inertia vs. number of clusters (k) to define the optimal *k*.
5. **Run K-Means Clustering model with the optimal *k***: Run the model with the optimal *k* to find the clusters.
6. **Visualization**: Plot the clusters in a 3D scatterplot, plot the XY and XZ projections to visualize the centroids, and visualize the clusters in the US map.
---
## Code Workflow
#### 1. Data Loading and Preprocessing
* Load the dataset from Kaggle with zipfile and os.
* Open the dataset with pandas.
* Aggregate data by county, averaging economic features, and drop NaN values.
```python
aggregated_data = data.groupby('County')[['IncomePerCap', 'Poverty', 'Unemployment']].mean().reset_index()
data_filtered = aggregated_data[['IncomePerCap', 'Poverty', 'Unemployment']].dropna()
```
#### 2. Preliminary Visual Analysis
* Use visualization libraries.
```python
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
```
* Graph three US counties map using geopandas and a geojson map, with colorscale to show unemplyment rate, poverty and GDP per capita.
```python
geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
counties_geo = gpd.read_file(geojson_url)
```
* Build a 3D scatterplot to visualize counties' distribution in terms of the three selected features.

#### 3. K-Means Clustering model
* Define a function to calculate the Euclidean distance.
* Define a function to assign clusters.
* Define a function to calculate new centroids.
* Define a function to run the model by consolidating the previous functions.
* Run K-Means model with a default k=4.

#### 4. Elbow Method
* Define a function to calculate inertia.
* Define *k* values to test (1 to 10).
* Calculate inertia for each *k* value.
* Plot the elbow curve.
* Visullay interpret the graph, find the elbow and choose optimal *k*.
#### 5. K-Means Clustering model with optimal *k*
* Run K-Means with k=3 clusters.
* Find the 3 clusters.

#### 6. Visualization
* See the clusters through a 3D scatterplot and the XY and XZ projections in the plane. Visualize the centroids of the three clusters.
* Plot the US counties map with the 3 clusters found through K-Means.

## Results
* Thanks to K-Means Clustering, we were able to identify three clusters for US counties according to three economic features (poverty rate, unemployment rate and income per capita). We can see that there are three clear groups:

    * Cluster 1: shows the best economic results
    * Cluster 2: shows medium results.
    * Cluster 3: shows more low-income counties.


