# single-cell-metabolomics
Visualization provides an ability to comprehend huge amounts of data.

Visualization allows the perception of emergent properties that were not anticipated.

A visualization commonly reveals things not only about the data per se, but about the way the data are collected. Visualization often enables problems with the data to become immediately apparent.

Visualization facilitates hypothesis formation.


Principal component analysis (PCA)

Widely used method for unsupervised, linear dimensionality reduction to maximize variance of data in as few dimensions as possible.

Caveats:
PCA linear projections often poorly represent relationships in multidimensional biological datasets.

Mostly preserves distances between dissimilar points, but directions of greatest variance may not be most informative.


t-SNE: t-Distributed Stochastic Neighbor Embedding

• Laurens van der Maaten and Geoffrey Hinton, 2008

• Widely used dimensionality reduction (biological and non-biological applications)

• Measure pairwise distances in high-dimensional space

• Move data points in low-dimensional space until the best position is found

• Minimize Kullback-Leibler Divergence 

• Points similar in high-D should be mapped close in low-D

In t-SNE, ‘best’ means that the points that are similar in HD will be mapped close to each other in LD space (t-SNE preserves local structure)

t-SNE is driven by 2 sets of forces: attractive forces bring similar datapoints closer to each other; repulsive forces push dissimilar datapoints away from each other

• We move datapoints in low-dimensional space until the best position is found:

• Minimization of Kullback-Leibler Divergence (KLD);

• KLD shows how much information is lost when the high dimensional data are flattened


UMAP

• How does it work?

• Looks for low dimensional topological representation

• How well does it work?

• Equally meaningful

representations compared with t-SNE

• Better representation of multi-branched continuous trajectories of hematopoietic development

• Faster than Barnes-Hut, comparable to FIt-SNE

