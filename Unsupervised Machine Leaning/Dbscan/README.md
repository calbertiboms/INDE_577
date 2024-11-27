# **DBScan**

## Overview 
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised machine learning algorithm that clusters data points based on density. Unlike K-Means, DBSCAN does not require specifying the number of clusters beforehand and is effective for handling noise and detecting clusters of arbitrary shapes.

In this project, we use DBSCAN to analyze earthquake data, grouping earthquake locations into clusters and identifying outliers based on geographic coordinates.

---

## **How DBSCAN Works**
1. **Core Points**: A point is a core point if it has at least `min_pts` neighbors within a radius `eps`.
2. **Density-Reachable Points**: Points within `eps` of a core point are part of the same cluster.
3. **Outliers (Noise)**: Points that are not reachable from any core point are labeled as noise.

**Key Advantages**:
- Automatically determines the number of clusters.
- Handles noise and works well for spatial data.
- Can identify clusters of arbitrary shapes.


### **DBSCAN vs K-Means**

| Feature              | DBSCAN                                | K-Means                            |
|----------------------|---------------------------------------|------------------------------------|
| **Cluster Shape**    | Arbitrary                            | Spherical                         |
| **Outliers**         | Labeled as noise                     | Included in clusters              |
| **Number of Clusters** | Determined automatically            | Must be specified in advance      |
| **Use Case**         | Ideal for spatial data with noise     | Works for well-separated clusters |


![DBSCAN vs K-Means](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*NsHxetYrFR_UZyhq)

---

## **Project Objectives**
1. **Load Earthquake Data**: Import a dataset containing earthquake events (`significant_earthquakes.csv`).
2. **Preprocess Data**: Extract geographic coordinates (latitude and longitude) for clustering.
3. **Apply DBSCAN**: Perform density-based clustering to group earthquakes and detect outliers.
4. **Visualize Results**: Overlay the clusters on a world map and interpret the findings.

---

## **Code Workflow**

### **1. Data Loading and Preprocessing**
- Load the dataset and extract latitude and longitude for clustering.
- Optionally standardize the data for better clustering performance.
```python
coordinates = df[['latitude', 'longitude']].to_numpy() 
```

### **2. DBSCAN Clustering**
DBSCAN parameters:
- `eps`: The maximum radius for neighborhood points.
- `min_pts`: Minimum points needed to form a dense cluster.

**Example Usage**:
```python
labels = dbscan(coordinates, eps=10, min_pts=5)
df['cluster'] = labels  # Add cluster labels to the DataFrame 
```

### **3. Visualization**
Visualize clusters on a world map using GeoPandas:

```python

geo_df[geo_df['cluster'] == cluster_id].plot(
    ax=ax, markersize=50, color=color, label=label
)
```
- Noise points are shown in black.
- Clusters are differentiated by unique colors.

--- 
## **Results**
- Clusters Detected: DBSCAN successfully identified distinct earthquake clusters and isolated noise points.
- Geographic Insights:
    - Clusters align with tectonic boundaries like the Pacific Ring of Fire and the Sunda Trench.
    - Noise points correspond to rare, isolated earthquakes.








