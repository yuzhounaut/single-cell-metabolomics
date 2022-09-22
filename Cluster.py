# Step 1: Import Libraries
#---------------------------------------------
# Data processing 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings("ignore")
# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import umap
import umap.plot
# Modeling
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import hdbscan
# Image output settings
from matplotlib import rcParams
# figure size in inches
# Width, height
rcParams['figure.figsize'] = 12,18
rcParams['figure.dpi'] = 300
#---------------------------------------------
# Step 2: Read Data
#---------------------------------------------
# Load data
#Creating dataframe for data.
df = pd.read_csv('D:/Download/iris-1.csv')
#Creating dataframe for Class.
#'D:/1808/Data/fds-20220516/Data/000A549/single cell_group.csv
Class = pd.read_csv('D:/Download/Species.csv')
Class['Class'].unique()
df['Class'] = Class
df['Class'].value_counts()
# Remove Class for the clustering model
X = df[df.columns.difference(['Class'])]
#Standardize the features before performing dimensionality reduction, (mean=0,standard deviation =1)
#X_std = StandardScaler().fit_transform(X)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
#---------------------------------------------
# Step 3: Decide the Number of Clusters
#---------------------------------------------
# please check out 
#https://medium.com/grabngoinfo/5-ways-for-deciding-number-of-clusters-in-a-clustering-model-5db993ea5e09
#---------------------------------------------
# Step 4: Kmeans Clustering (Model 1)
#---------------------------------------------
# Kmeans model
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
# Fit and predict on the data
y_kmeans = kmeans.fit_predict(X_std)
# Save the predictions as a column
df['y_kmeans']=y_kmeans
# Check the distribution
df['y_kmeans'].value_counts()
print(df['y_kmeans'].value_counts())
#---------------------------------------------
# Step 5: Hierarchical Clustering (Model 2)
#---------------------------------------------
# Hierachical clustering model
hc = AgglomerativeClustering(n_clusters = 3)
# Fit and predict on the data
y_hc = hc.fit_predict(X_std)
# Save the predictions as a column
df['y_hc']=y_hc
# Check the distribution
df['y_hc'].value_counts()
print(df['y_hc'].value_counts())
#---------------------------------------------
# Step 6: Gaussian Mixture Model (GMM) (Model 3)
#---------------------------------------------
# Fit the GMM model
gmm = GaussianMixture(n_components=3, n_init=5, random_state=42)
# Fit and predict on the data
y_gmm = gmm.fit_predict(X_std)
# Save the prediction as a column
df['y_gmm']=y_gmm
# Check the distribution
df['y_gmm'].value_counts()
print(df['y_gmm'].value_counts())
#---------------------------------------------
# Step 7: Density-based spatial clustering of applications with noise (DBSCAN) (Model 4)
#---------------------------------------------
# Fit the DBSCAN model
dbscan = DBSCAN(eps=0.5, min_samples=5) 
# Fit and predict on the data
y_dbscan = dbscan.fit_predict(X_std)
# Save the prediction as a column
df['y_dbscan'] = y_dbscan
# Check the distribution
df['y_dbscan'].value_counts()
print(df['y_dbscan'].value_counts())
#Fit the HDBSCAN model
hdbscan = hdbscan.HDBSCAN(min_samples=15, min_cluster_size=30)
# Fit and predict on the data
y_hdbscan = hdbscan.fit_predict(X_std)
# Save the prediction as a column
df['y_hdbscan'] = y_hdbscan
# Check the distribution
df['y_hdbscan'].value_counts()
print(df['y_hdbscan'].value_counts())
#---------------------------------------------
# Step 8: Dimensionality Reduction
#---------------------------------------------
# PCA with 2 components
pca=PCA(n_components=2).fit_transform(X_std)
# Create columns for the 2 PCA components
df['PC1'] = pca[:, 0]
df['PC2'] = pca[:, 1]

# TSNE with 2 components
tsne=TSNE(learning_rate = 10, n_components = 2, random_state = 42, perplexity = 30).fit_transform(X_std)
# Create columns for the 2 TSNE components
df['t-SNE1'] = tsne[:, 0]
df['t-SNE2'] = tsne[:, 1]

# UMAP with 2 components
umapper=UMAP(n_neighbors=50, min_dist=0.4, n_components=2, random_state=42).fit_transform(X_std)
# Create columns for the 2 TSNE components
df['UMAP1'] = umapper[:, 0]
df['UMAP2'] = umapper[:, 1]

# Take a look at the data
df.head()
print(df.head())
#---------------------------------------------
# Step 9: Visual Comparison of Models
#---------------------------------------------
# Check label mapping
df.groupby(['Class', 'y_kmeans']).size().reset_index(name='counts')
print(df.groupby(['Class', 'y_kmeans']).size().reset_index(name='counts'))
# Rename Class
df['y_kmeans'] = df['y_kmeans'].map({1: 0, 2: 1, 0: 2})

# Check label mapping
df.groupby(['Class', 'y_hc']).size().reset_index(name='counts')
print(df.groupby(['Class', 'y_hc']).size().reset_index(name='counts'))
# Rename Class
df['y_hc'] = df['y_hc'].map({1: 0, 2: 1, 0: 2})

# Check label mapping
df.groupby(['Class', 'y_gmm']).size().reset_index(name='counts')
print(df.groupby(['Class', 'y_gmm']).size().reset_index(name='counts'))
# Rename Class
df['y_gmm'] = df['y_gmm'].map({1: 0, 2: 1, 0: 2})

# Check label mapping
df.groupby(['Class', 'y_dbscan']).size().reset_index(name='counts')
print(df.groupby(['Class', 'y_dbscan']).size().reset_index(name='counts'))
# Rename Class
df['y_dbscan'] = df['y_dbscan'].map({0: 0, -1: 2, 1: 1})

# Check label mapping
df.groupby(['Class', 'y_hdbscan']).size().reset_index(name='counts')
print(df.groupby(['Class', 'y_hdbscan']).size().reset_index(name='counts'))
# Rename Class
df['y_hdbscan'] = df['y_hdbscan'].map({0: 0, -1: 2, 1: 1})

# Visualization using PCA
palette = sns.color_palette("Set2", n_colors=3)
#Attention: the number of colors is the class number
sns.set_style("white")
fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(12,18))
sns.scatterplot(x='PC1', y='PC2', data=df, hue='Class', palette=palette, linewidth=0.2, s=30, 
                alpha=0.8, ax=axs[0,0]).set(title='Ground Truth')
sns.scatterplot(x='PC1', y='PC2', data=df, hue='y_kmeans', palette=palette, linewidth=0.2, s=30, 
                alpha=0.8, ax=axs[0,1]).set(title='KMeans')
sns.scatterplot(x='PC1', y='PC2', data=df, hue='y_hc', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[1,0]).set(title='Hierachical')
sns.scatterplot(x='PC1', y='PC2', data=df, hue='y_gmm', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[1,1]).set(title='GMM')
sns.scatterplot(x='PC1', y='PC2', data=df, hue='y_dbscan', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[2,0]).set(title='DBSCAN')
sns.scatterplot(x='PC1', y='PC2', data=df, hue='y_hdbscan', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[2,1]).set(title='HDBSCAN')

# Visualization using t-SNE
fig, axs = plt.subplots(nrows=3, ncols=2, sharey=True, figsize=(12,18))
sns.scatterplot(x='t-SNE1', y='t-SNE2', data=df, hue='Class', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[0,0]).set(title='Ground Truth')
sns.scatterplot(x='t-SNE1', y='t-SNE2', data=df, hue='y_kmeans', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[0,1]).set(title='KMeans')
sns.scatterplot(x='t-SNE1', y='t-SNE2', data=df, hue='y_hc', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[1,0]).set(title='Hierachical')
sns.scatterplot(x='t-SNE1', y='t-SNE2', data=df, hue='y_gmm', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[1,1]).set(title='GMM')
sns.scatterplot(x='t-SNE1', y='t-SNE2', data=df, hue='y_dbscan', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[2,0]).set(title='DBSCAN')
sns.scatterplot(x='t-SNE1', y='t-SNE2', data=df, hue='y_hdbscan', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[2,1]).set(title='HDBSCAN')

# Visualization using UMAP
fig, axs = plt.subplots(nrows=3, ncols=2, sharey=True, figsize=(12,18))
sns.scatterplot(x='UMAP1', y='UMAP2', data=df, hue='Class', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[0,0]).set(title='Ground Truth')
sns.scatterplot(x='UMAP1', y='UMAP2', data=df, hue='y_kmeans', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[0,1]).set(title='KMeans')
sns.scatterplot(x='UMAP1', y='UMAP2', data=df, hue='y_hc', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[1,0]).set(title='Hierachical')
sns.scatterplot(x='UMAP1', y='UMAP2', data=df, hue='y_gmm', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[1,1]).set(title='GMM')
sns.scatterplot(x='UMAP1', y='UMAP2', data=df, hue='y_dbscan', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[2,0]).set(title='DBSCAN')
sns.scatterplot(x='UMAP1', y='UMAP2', data=df, hue='y_hdbscan', palette=palette, linewidth=0.2, s=30, alpha=0.8,
                ax=axs[2,1]).set(title='HDBSCAN')



