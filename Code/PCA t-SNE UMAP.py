import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import umap
import umap.plot
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 6,6
rcParams['figure.dpi'] = 300

#Creating dataframe for data.
data = pd.read_csv('D:/t-SNE UMAP/Data/UMAP.csv', delimiter=',',index_col=0)
nRow, nCol = data.shape
print(f'There are {nRow} rows and {nCol} columns in data.')

#Inspect first 5 rows of data.
data.head(5)
print(data.head(5))

#Creating dataframe for labels.
labels = pd.read_csv('D:/t-SNE UMAP/Data/UMAP_group.csv', delimiter=',')
nRow, nCol = labels.shape
print(f'There are {nRow} rows and {nCol} columns in labels dataframe.')

#Inspect first 5 rows of labels.
labels.head(5)
print(labels.head(5))

#Find unique classes of cell types.
labels['Class'].unique()
print(labels['Class'].unique())

#Create a 2D numpy array of values in data.
X = data.values
X[0:5]

#Standardize the features before performing dimensionality reduction, (mean=0,standard deviation =1)
X_std = StandardScaler().fit_transform(X)

#Visualize data using Principal Component Analysis.
print("Principal Component Analysis (PCA)")
pca = PCA(n_components = 2).fit_transform(X_std)
pca_df = pd.DataFrame(data=pca, columns=['PC1','PC2']).join(labels)
plt.figure(dpi=300, figsize = (6,6))
palette = sns.color_palette("Set2", n_colors=3)
#Attention: the number of colors is the class number
sns.set_style("white")
sns.scatterplot(x='PC1',y='PC2',hue='Class',data=pca_df, palette=palette, linewidth=0.2, s=30, alpha=0.8).set_title('PCA')

#Fitting PCA on Data
print("Explained Variance of PCA components")
pca_std = PCA().fit(X_std)
percent_variance=pca_std.explained_variance_ratio_*100

#Plotting Cumulative Summation of the Explained Variance
plt.figure(dpi=300)
plt.plot(np.cumsum(pca_std.explained_variance_ratio_), marker = '.', linestyle = '--')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Cumulative Explained Variance')
plt.show()

#Visualize data using t-SNE.
print("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
model = TSNE(learning_rate = 10, n_components = 2, random_state= 42, perplexity = 30)
tsne = model.fit_transform(X_std, figsize = (6,6))
tsne_df = pd.DataFrame(data=tsne, columns=['t-SNE1','t-SNE2']).join(labels)
plt.figure(dpi=300)
palette = sns.color_palette("Set2", n_colors=3)
#Attention: the number of colors is the class number
sns.set_style("white")
sns.scatterplot(x='t-SNE1',y='t-SNE2',hue='Class',data=tsne_df, palette=palette, linewidth=0.2, s=30, alpha=0.8).set_title('t-SNE')

#Measure execution time for t-SNE
def tsne_model(X):
    model = TSNE(learning_rate = 10, n_components = 2, random_state = 42, perplexity = 30)
    tsne = model.fit_transform(X)
    return tsne
from timeit import Timer
  
t = Timer(lambda: tsne_model(X_std))
print(t.timeit(number=1))

#Visualize data using t-SNE after PCA.
print("t-Distributed Stochastic Neighbor Embedding (t-SNE) on PCA")
#Attention: For large datasets, keeping the first 100 components, rather than the usual 50 to minmize distortion from the initial PCA.
X_reduced = PCA(n_components = 100).fit_transform(X_std)
model = TSNE(learning_rate = 10, n_components = 2, random_state = 42, perplexity = 30)
tsne_pca = model.fit_transform(X_reduced)
tsne_pca_df = pd.DataFrame(data=tsne_pca, columns=['t-SNE1','t-SNE2']).join(labels)
plt.figure(dpi=300, figsize = (6,6))
palette = sns.color_palette("Set2", n_colors=3)
#Attention: the number of colors is the class number
sns.set_style("white")
sns.scatterplot(x='t-SNE1',y='t-SNE2',hue='Class',data=tsne_pca_df, palette=palette, linewidth=0.2, s=30, alpha=0.8).set_title('t-SNE after PCA')

#Measure execution time for t-SNE after PCA
def tsne_model_pca(X):
    X_reduced = PCA(n_components = 100).fit_transform(X_std)
    model = TSNE(learning_rate = 10, n_components = 2, random_state = 42, perplexity = 30)
    tsne_pca = model.fit_transform(X_reduced)
    return tsne_pca
from timeit import Timer
  
t = Timer(lambda: tsne_model_pca(X_std))
print(t.timeit(number=1))

#Visualize data using UMAP.
print("Uniform Manifold Approximation and Projection (UMAP)")
model = UMAP(n_neighbors = 50, min_dist = 0.4, n_components = 2)
umapper = model.fit_transform(X_std)
umap_df = pd.DataFrame(data=umapper, columns=['UMAP1','UMAP2']).join(labels)
plt.figure(dpi=300, figsize = (6,6))
palette = sns.color_palette("Set2", n_colors=3)
#Attention: the number of colors is the class number
sns.set_style("white")
sns.scatterplot(x='UMAP1',y='UMAP2',hue='Class',data=umap_df, palette=palette, linewidth=0.2, s=30, alpha=0.8).set_title('UMAP')

umap_df
mapper = UMAP(n_neighbors = 50, min_dist = 0.4, n_components = 2).fit(X)
plt.figure(dpi=300,figsize = (6,6))
#dpi value times figure size values in inches equals to pixel values 6*300=1800
umap.plot.points(mapper, labels=umap_df.Class, color_key_cmap='Set2', width=1800, height=1800)

umap.plot.connectivity(mapper, show_points=True, width=1800, height=1800)
umap.plot.connectivity(mapper, edge_bundling='hammer', width=1800, height=1800)

umap.plot.diagnostic(mapper, diagnostic_type='pca', point_size=30, width=1800, height=1800)
umap.plot.diagnostic(mapper, diagnostic_type='local_dim', point_size=30, width=1800, height=1800)
umap.plot.diagnostic(mapper, diagnostic_type='neighborhood', point_size=30, width=1800, height=1800)

#Measure execution time for UMAP
def umap_model(X):
    model = UMAP(n_neighbors = 50, min_dist = 0.4, n_components = 2)
    umapvar = model.fit_transform(X)
    return umapvar
from timeit import Timer
  
t = Timer(lambda: umap_model(X_std))
print(t.timeit(number=1))


