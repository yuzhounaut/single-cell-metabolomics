import numpy as np
import pandas as pd

DATASET_PATH=r'2000cellsPA20220407 all peaks hepg2.csv'
cell_data_df= pd.read_csv(DATASET_PATH,low_memory=False)
cell_data_df.head()
cell_data_df.columns
#specify the 990 features column names to be modelled
to_model_columns=cell_data_df.columns[1:990]

from sklearn.ensemble import IsolationForest
clf=IsolationForest(n_estimators=100, max_samples='auto', \
                        max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
clf.fit(cell_data_df[to_model_columns])

pred = clf.predict(cell_data_df[to_model_columns])
cell_data_df['anomaly']=pred
outliers=cell_data_df.loc[cell_data_df['anomaly']==-1]
outlier_index=list(outliers.index)
#print(outlier_index)
#Find the number of anomalies and normal points here points classified -1 are anomalous
print(cell_data_df['anomaly'].value_counts())
print(outliers)
outliers.to_csv(r'Outliers HepG2 32 in 1000-052311.csv')

#3D PCA
#HepG2 #66c2a5
#MCF7 #fc8d62
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
pca = PCA(n_components=3)  # Reduce to k=3 dimensions
scaler = StandardScaler()
#normalize the cell_data
X = scaler.fit_transform(cell_data_df[to_model_columns])
X_reduce = pca.fit_transform(X)

figsize=(6, 6)
fig = plt.figure(figsize=figsize, dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.set_zlabel("x_composite_3")

# Plot the compressed data points
ax.scatter(X_reduce[:, 0], X_reduce[:, 1], zs=X_reduce[:, 2], s=8, lw=1, label="Normal",c="#a6d854")

# Plot x's for the ground truth outliers
ax.scatter(X_reduce[outlier_index,0],X_reduce[outlier_index,1], X_reduce[outlier_index,2],
           lw=2, s=33, marker="x", c="red", label="Outliers")
ax.legend()
plt.show()

#2D PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca2 = PCA(n_components=2)# Reduce to k=2 dimensions
scaler = StandardScaler()
#normalize the cell_data
XT = scaler.fit_transform(cell_data_df[to_model_columns])
pca2.fit(XT)
res=pd.DataFrame(pca2.transform(XT))

Z = np.array(res)
figsize=(8, 8)
plt.figure(figsize=figsize, dpi=300)
plt.title("Isolation Forest")
#plt.contourf(Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(res[0], res[1], c='#a6d854', s=33,label="Normal")

b1 = plt.scatter(res.iloc[outlier_index,0],res.iloc[outlier_index,1], c='red',
                 s=33, edgecolor="red",label="Outliers")
plt.legend(loc="upper right")
plt.show()
