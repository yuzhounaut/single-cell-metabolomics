# Single-cell-metabolomics
Visualization provides an ability to comprehend huge amounts of data.

Visualization allows the perception of emergent properties that were not anticipated.

A visualization commonly reveals things not only about the data per se, but about the way the data are collected. Visualization often enables problems with the data to become immediately apparent.

Visualization facilitates hypothesis formation.




* **Principal component analysis (PCA)**

Widely used method for unsupervised, linear dimensionality reduction to maximize variance of data in as few dimensions as possible.

 **_Caveats:_**
PCA linear projections often poorly represent relationships in multidimensional biological datasets.

Mostly preserves distances between dissimilar points, but directions of greatest variance may not be most informative.
![Figure 2021-06-24 182515 PCA](https://user-images.githubusercontent.com/86154919/123248468-8d372700-d51a-11eb-81cf-395614d3000b.png)
![Figure 2021-06-24 183017 Cumulative Explained Variance](https://user-images.githubusercontent.com/86154919/123248485-91634480-d51a-11eb-9cbf-4f453305ce96.png)





* **t-SNE: t-Distributed Stochastic Neighbor Embedding**

  * Laurens van der Maaten and Geoffrey Hinton, 2008

  * Widely used dimensionality reduction (biological and non-biological applications)

  * Measure pairwise distances in high-dimensional space

  * Move data points in low-dimensional space until the best position is found

  * Minimize Kullback-Leibler Divergence 

  * Points similar in high-D should be mapped close in low-D

In t-SNE, ‘best’ means that the points that are similar in HD will be mapped close to each other in LD space (t-SNE preserves local structure)

t-SNE is driven by 2 sets of forces: attractive forces bring similar datapoints closer to each other; repulsive forces push dissimilar datapoints away from each other

  * We move datapoints in low-dimensional space until the best position is found:

  * Minimization of Kullback-Leibler Divergence (KLD);

  * KLD shows how much information is lost when the high dimensional data are flattened
![Figure 2021-06-24 183031 t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://user-images.githubusercontent.com/86154919/123248510-988a5280-d51a-11eb-91a5-e58a93f2d0f1.png)
![Figure 2021-06-24 183159 t-Distributed Stochastic Neighbor Embedding (t-SNE) on PCA](https://user-images.githubusercontent.com/86154919/123248523-9c1dd980-d51a-11eb-960b-d62770dae8b6.png)





* **UMAP**

How does it work?

  * Looks for low dimensional topological representation

How well does it work?

  * Equally meaningful

Compared with t-SNE

  * Better representation of multi-branched continuous trajectories of hematopoietic development

  * Faster than Barnes-Hut, comparable to FIt-SNE
 ![Figure 2021-06-24 183212 UMAP](https://user-images.githubusercontent.com/86154919/123248551-a213ba80-d51a-11eb-9ea8-67332373fc6c.png)

_**These data visualization results are obtained using the same data but different methods.**_
Multiple clustering methods and visualizations
![Figure 2022-09-22 093738](https://user-images.githubusercontent.com/86154919/191640069-055a19b1-92f2-48a2-a815-3ac402d72077.png)
![Figure 2022-09-22 093741](https://user-images.githubusercontent.com/86154919/191640083-e38f6550-4825-494a-b1c0-96a2868a2df0.png)
![Figure 2022-09-22 093745](https://user-images.githubusercontent.com/86154919/191640115-6d9b79e3-495b-4cb9-9a6e-3fb816bcfb3c.png)
