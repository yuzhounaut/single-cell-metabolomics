import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import umap
import umap.plot

from umap.umap_ import nearest_neighbors
import leidenalg
from scanpy.neighbors import _compute_connectivities_umap
from scanpy._utils import get_igraph_from_adjacency
#https://github.com/MiqG/leiden_clustering
#Leiden Clustering
#Infers nearest neighbors graph from data matrix and uses the Leiden algorithm
#to cluster the observations into clusters or communities.
#Specifically, (i) compress information with PCA, (ii) compute nearest
#neighbors graph with UMAP, (iii) cluster graph with leiden algorithm.
class LeidenClustering:
    """
    Leiden Clustering
    
    Infers nearest neighbors graph from data matrix and uses the Leiden algorithm
    to cluster the observations into clusters or communities.
    Specifically, (i) compress information with PCA, (ii) compute nearest
    neighbors graph with UMAP, (iii) cluster graph with leiden algorithm.
    
    This is a class wrapper based on https://github.com/theislab/scanpy/blob/c488909a54e9ab1462186cca35b537426e4630db/scanpy/tools/_leiden.py.
    
    Parameters
    ----------
    pca_kws : dict, default={"n_components":10}
        Parameters to control PCA step using `sklearn.decomposition.PCA`.
        
    nn_kws :  dict, default={"n_neighbors": 30, "metric": "cosine", 
    "metric_kwds": {}, "angular": False, "random_state": np.random}
        Parameters to control generation of nearest neighbors graph using 
        `umap.umap_.nearest_neighbors`.
    
    partition_type : type of `class`, default=leidenalg.RBConfigurationVertexPartition
        The type of partition to use for optimization of the Leiden algorithm.
        
    leiden_kws : dict, default={"n_iterations": -1, "seed": 0}
        Parameters to control Leiden algorithm using `leidenalg.find_partition`.
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each data point.
    
    Examples
    --------
    from leiden_clustering import LeidenClustering
    import numpy as np
    X = np.random.randn(100,10)
    clustering = LeidenClustering()
    clustering.fit(X)
    clustering.labels_
    """
    def __init__(
        self,
        pca_kws={"n_components":100},
        nn_kws={
            "n_neighbors": 50,
            "metric": "cosine",
            "metric_kwds": {},
            "angular": False,
            "random_state": 42,
        },
        partition_type=leidenalg.RBConfigurationVertexPartition,
        leiden_kws={"n_iterations": -1, "seed": 0},
    ):
        self.pca_kws = pca_kws
        self.nn_kws = nn_kws
        self.partition_type = partition_type
        self.leiden_kws = leiden_kws
        
    def fit(self, X):
        # compress information with PCA
        pca = PCA(**self.pca_kws)
        pcs = pca.fit_transform(X)
        
        # compute nearest neighbors with UMAP
        knn_indices, knn_dists, forest = nearest_neighbors(pcs, **self.nn_kws)

        # compute connectivities
        distances, connectivities = _compute_connectivities_umap(
            knn_indices, knn_dists, pcs.shape[0], self.nn_kws["n_neighbors"]
        )

        # use connectivites as adjacency matrix to get igraph
        G = get_igraph_from_adjacency(connectivities, directed=True)

        # run leiden on graph
        self.leiden_kws["weights"] = np.array(G.es["weight"]).astype(np.float64)

        partition = leidenalg.find_partition(G, self.partition_type, **self.leiden_kws)
        labels = np.array(partition.membership)
        
        self.labels_ = labels
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_


#Creating dataframe for data.
data = pd.read_csv('D:/1808/Data/fds-20220516/Data/075A549/075A549-20220831 norm for plot no g.csv', 
                   delimiter=',', index_col=0)
nRow, nCol = data.shape
print(f'There are {nRow} rows and {nCol} columns in data.')

#Inspect first 5 rows of data.
data.head(5)
print(data.head(5))

#Create a 2D numpy array of values in data.
X = data.values
X[0:5]

#Standardize the features (mean=0,standard deviation =1)
X_std = StandardScaler().fit_transform(X)

clustering = LeidenClustering()
clustering.fit(X_std)
clustering.labels_
#generate data_label dict
cluster_label = {'Cluster': pd.Series(data=clustering.labels_)}
#generate df_label dataframe
clabels = pd.DataFrame(cluster_label)
#generate d_label dataframe (data with lables)
d = data.reset_index(drop=False)
d_lable = pd.DataFrame(data=d).join(clabels)
d_lable.to_csv('D:/Download/000A549_d_lable.csv', index=False)

#PCA with error ellipses
scaler = StandardScaler()
data_std = scaler.fit_transform(data)
pca = PCA(n_components = 2).fit_transform(data_std)
pca_df = pd.DataFrame(data=pca, columns=['PC1','PC2']).join(clabels)

plt.figure(dpi=300,figsize = (6,6))
sns.set_style("white")
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue = 'Cluster', palette = 'Set2',
                linewidth=0.2, s=30, alpha=0.8)
plt.title("Leiden Clustering with PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")

#t-SNE
tSNEdata = TSNE(n_components=2, learning_rate=10, random_state=42, perplexity=30).fit_transform(data_std)
tSNE_df = pd.DataFrame(data=tSNEdata, columns=['t-SNE1','t-SNE2']).join(clabels)
plt.figure(dpi=300,figsize = (6,6))
sns.set_style("white")
sns.scatterplot(data=tSNE_df, x="t-SNE1", y="t-SNE2", hue = 'Cluster', palette = 'Set2',
                linewidth=0.2, s=30, alpha=0.8)
plt.title("Leiden Clustering with t-SNE")
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")

#UMAP
umapdata = UMAP(n_neighbors=50, min_dist=0.4, n_components=2, random_state=42).fit_transform(data_std)
umap_df = pd.DataFrame(data=umapdata, columns=['UMAP1','UMAP2']).join(clabels)
plt.figure(dpi=300,figsize = (6,6))
sns.set_style("white")
sns.scatterplot(data=umap_df, x="UMAP1", y="UMAP2", hue = 'Cluster', palette = 'Set2',
                linewidth=0.2, s=30, alpha=0.8)
plt.title("Leiden Clustering with UMAP")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")



