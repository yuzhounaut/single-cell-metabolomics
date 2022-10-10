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
data = pd.read_csv('D:/1808/Data/fds-20220516/Data/075A549/075A549-20220831 norm for plot no g.csv', 
                   delimiter=',',index_col=0)
nRow, nCol = data.shape
print(f'There are {nRow} rows and {nCol} columns in data.')

#Inspect first 5 rows of data.
data.head(5)
print(data.head(5))

#Creating dataframe for labels.
labels = pd.read_csv('D:/1808/Data/fds-20220516/Data/075A549/single cell_group no g.csv', delimiter=',')
nRow, nCol = labels.shape
print(f'There are {nRow} rows and {nCol} columns in labels dataframe.')

#Inspect first 5 rows of labels.
labels.head(5)
print(labels.head(5))

#Find unique classes of cell types.
labels['Class'].unique()
print(labels['Class'].unique())
#Get the number of unique cell types.
unique_value = labels['Class'].nunique()
print(unique_value)

#Create a 2D numpy array of values in data.
X = data.values
X[0:5]

#Standardize the features before performing dimensionality reduction, (mean=0,standard deviation =1)
X_std = StandardScaler().fit_transform(X)

#Fitting PCA on Data
print("Explained Variance of PCA components")
pca_std = PCA().fit(X_std)
percent_variance=pca_std.explained_variance_ratio_*100

#Plotting Cumulative Summation of the Explained Variance
plt.figure(dpi=300, figsize = (6,6))
plt.plot(np.cumsum(pca_std.explained_variance_ratio_), marker = '.', linestyle = '--')
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Cumulative Explained Variance')
plt.show()

"""
A confidence ellipse is an ellipse which is concentric to the error ellipse.
 chi-square distribution 
 
The error ellipse is a confidence ellipse with elliptical scale factor k = 1 
and probability approximately p = 0.3935. The 50% and 95% confidence ellipses have 
elliptical scale factors approximately 1.1774 and 2.4477, respectively.
First, the error ellipse
"""
#N-sigma error ellipse for PCA plot
from matplotlib.patches import Ellipse
def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

#Visualize data using Principal Component Analysis.
print("Principal Component Analysis (PCA)")
pca = PCA(n_components = 2).fit_transform(X_std)

#Set2 palette hex code
plt.figure(dpi=300, figsize = (6,6))
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
pca_df = pd.DataFrame(data=pca, columns=['PC1','PC2']).join(labels)
sns.set_style("white")
n_colors = unique_value
palette = sns.color_palette("Set2", n_colors=n_colors)
#Attention: the number of colors is the class number
lbs = labels['Class'].unique()
for i, l in enumerate(lbs):
    k = np.where(labels == l)[0]
    pts = pca[k,:]
    # Plot the raw points...    #.T is np.transpose()
    x, y = pts.T
    # Plot a transparent 2 standard deviation covariance ellipse
    plot_point_cov(pts, nstd=2, alpha=0.2, color=colors[i])
sns.scatterplot(x='PC1', y='PC2', hue='Class', data=pca_df, palette=palette, linewidth=0.2, s=30, 
                alpha=0.8).set_title('PCA')
plt.title('PCA')   
plt.xlabel('PC1 ({} %)'.format(round(pca_std.explained_variance_ratio_[0] * 100, 2)))
plt.ylabel('PC2 ({} %)'.format(round(pca_std.explained_variance_ratio_[1] * 100, 2)))
plt.show()

#95% confidence ellipse for PCA plot




#Visualize data using t-SNE.
print("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
model = TSNE(learning_rate = 10, n_components = 2, random_state = 42, perplexity = 30)
tsne = model.fit_transform(X_std)
tsne_df = pd.DataFrame(data=tsne, columns=['t-SNE1','t-SNE2']).join(labels)
plt.figure(dpi=300,figsize = (6,6))
palette = sns.color_palette("Set2", n_colors=n_colors)
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
plt.figure(dpi=300,figsize = (6,6))
palette = sns.color_palette("Set2", n_colors=n_colors)
#Attention: the number of colors is the class number
sns.set_style("white")
sns.scatterplot(x='t-SNE1',y='t-SNE2',hue='Class',data=tsne_pca_df, palette=palette, linewidth=0.2, s=30, alpha=0.8).set_title('t-SNE after PCA')

#Measure execution time for t-SNE after PCA
def tsne_model_pca(X):
    X_reduced = PCA(n_components = 100 ).fit_transform(X_std)
    model = TSNE(learning_rate = 10, n_components = 2, random_state = 42, perplexity = 30)
    tsne_pca = model.fit_transform(X_reduced)
    return tsne_pca
from timeit import Timer
  
t = Timer(lambda: tsne_model_pca(X_std))
print(t.timeit(number=1))

#Visualize data using UMAP.
print("Uniform Manifold Approximation and Projection (UMAP)")
model = UMAP(n_neighbors = 50, min_dist = 0.4, n_components = 2, random_state=42)
umapper = model.fit_transform(X_std)
umap_df = pd.DataFrame(data=umapper, columns=['UMAP1','UMAP2']).join(labels)
plt.figure(dpi=300, figsize = (6,6))
palette = sns.color_palette("Set2", n_colors=n_colors)
#Attention: the number of colors is the class number
sns.set_style("white")
sns.scatterplot(x='UMAP1',y='UMAP2',hue='Class',data=umap_df, palette=palette, linewidth=0.2, s=30, alpha=0.8).set_title('UMAP')

umap_df
mapper = UMAP(n_neighbors = 50, min_dist = 0.4, n_components = 2).fit(X_std)
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




