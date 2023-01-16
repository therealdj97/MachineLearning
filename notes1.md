# ML Algorithms

## Supervised Learning

### Tree Based

1. Decision Tree
Decision Tree models make decision rules on the features to produce predictions. It can be used for classification or regression

2. Random Forest
An ensemble learning method that combines the output of multiple decision trees

3. Gradient Boosting Regression
Gradient Boosting Regression employs boosting to make predictive models from an ensemble of weak predictive learners

4. XGBoost
Gradient Boosting algorithm that is eOcient & flexible. Can be used for both classification and regression tasks

5. LightGBM Regression
A gradient boosting framework that is designed to be more effcient than other implementations

### Linear Models

1. Linear Regression
A simple algorithm that models a linear relationship between inputs and a continuous numerical output variable

2. Logistic Regression
A simple algorithm that models a linear relationship between inputs and a categorical output (1 or 0)

3. Ridge Regression
Part of the regression family — it penalizes features that have low predictive outcomes by shrinking their coefficients closer to zero. Can be used for classification or regression

4. Lasso Regression
Part of the regression family — it penalizes features that have low predictive outcomes by shrinking their coeOcients to zero. Can be used for classification or regression

### Regression Only

1. Linear Regression
Linear Regression models a linear relationship between input variables and a continuous numerical output variable. The default loss function is the mean square error (MSE)

2. Polynomial regression
Polynomial Regression models nonlinear relationships between the dependent, and independent variable as the n-th degree polynomial.

3. Support vector Regression
Support Vector Regression (SVR) uses the same principle as SVMs but optimiRes the cost function to fit the most straight line (or plane) through the data points. With the kernel trick it can efficiently perform a non-linear regression by implicitly mapping their inputs into high-dimensional feature spaces.

4. Gausian Process Regression
Gaussian Process Regression (GPR) uses a Bayesian approach that infers a probability distribution over the possible functions that fit the data. The Gaussian process is a prior that is specified as a multivariate Gaussian distribution

5. Robust Regression
Robust Regression is an alternative to least squares regression when data is contaminated with outliers. The term “robust” refers to the statistical capability to provide useful information even in the face of outliers.

### Both regression / classification models

1. Decision Trees
Decision Tree models learn on the data by making decision rules on the variables to separate the classes in a flowchart like a tree data structure. They can be used for both regression and classification.

2. Random Forest
Random Forest classification models learn using an ensemble of decision trees. The output of the random forest is based on a majority vote of the different decision trees

3. Gradient Boosting
An ensemble learning method where weak predictive learners are combined to improve accuracy. Popular techniques include XGBoost, LightGBM and more.

4. Ridge Regression
Ridge Regression penaliRes variables with low predictive outcomes by shrinking their coefficients towards Rero. It can be used for classification and regression.

5. Lasso Regression
Lasso Regression penaliRes features that have low predictive outcomes  
by shrinking their coefficients to Rero. It can be used for classification  
and regression.
6. AdaBoost
Adaptive Boosting uses an ensemble of weak learners that is combined into a weighted sum that represents the final output of the boosted classifier.

* Advantages

1. Explainable & Interpretable
2. Less need for tweaking parameters
3. Usually outperforms Random Forest

* Disadvantages

1. Less prone to overfitting as the input variables are not jointly optimiRed@
2. ensitive to noisy data and outliers.

### Classification

1. SVM
In its simplest form, support vector machine is a linear classifier. But with the kernel trick, it can efficiently perform a non-linear classification by implicitly mapping their inputs into high-dimensional feature spaces. This makes SVM one of the best prediction methods.

2. Nearest Neighbors
Nearest Neighbors predicts the label based on a predefined number of samples closest in distance to the new point.

3. Logistic Regresssion (& its extensions)
The logistic regression models a linear relationship between input variables and the response variable. It models the output as binary values (0 or 1) rather than numeric values.

4. Linear Descriminent Analysis
The linear decision boundary maximiRes the separability between the classes by finding a linear combination of features

## Unsupervised Learning

### Association

1. Apriori
Rule based approach that identifies the most frequent itemset in a given dataset where prior knowledge of frequent itemset properties is used
The Apriori algorithm uses the join and prune step iteratively to identify the most frequent itemset in the given dataset. Prior knowledge (apriori) of frequent itemset properties is used in the process.

2. FP_Growth algorithm
Frequent Pattern growth (FP-growth) is an improvement on the Apriori algorithm for finding frequent itemsets. It generates a conditional FP-Tree for every item in the data

3. FP-Max Algorithm
A variant of Frequent pattern growth that is focused on finding maximal itemsets

4. Eclat
Equivalence Class Clustering and Bottom-Up Lattice Traversal (Eclat) applies a Depth First Search of a graph procedure. This is a more efficient and scalable version of the 
Apriori algorithm.

5. Hypergeometric Networks
HNet learns the Association from datasets with mixed data types (discrete and continuous variables) and with unknown functions. Associations are statistically tested using the hypergeometric distribution for finding frequent itemset.

### Clustering

1. K-Means
KMeans is the most widely used clustering approach—it determines K clusters based on euclidean distances
Most common clustering approach which assumes that the closer data points are to each other, the more similar they are. It determines K clusters based on Euclidean distances.

2. Hierarchical clustering
A bottomlup approach where each data point is treated as its own cluster—and then the closest two clusters are merged together iteratively

3. Gausian Mixture
A bottomlup approach where each data point is treated as its own cluster—and then the closest two clusters are merged together iteratively

4. DBSCAN
Density-Based Spatial Clustering of Applications with Noise can handle non-linear cluster structures, purely based on density. It can differentiate and separate regions with varying degrees of density, thereby creating clusters.

5. HDBSCAN
Family of the density-based algorithms and has roughly two steps: finding the core distance of each point, and expands clusters from them. It extends DBSCAN by converting it into a hierarchical clustering algorithm

6. Agglomerative Hierarchical Clustering
Uses hierarchical clustering to determine the distance between samples based on the metric, and pairs are merged into clusters using the linkage type.

7. OPTICS
Family of the density-based algorithms where it finds core sample of high density and expands clusters from them. It operates with a core distance (ɛ) and reachability distance

8. Gausian Mixture Models
Gaussian Mixture Models (GMM) leverages probabilistic models to detect clusters using a mixture of normal (gaussian) distributions.

### Dimensionality Reduction

1. PCA
Principal Component Analysis (PCA) is a feature extraction approach that uses a linear function to reduce dimensionality in datasets by minimizing information loss.

2. t-SNE
t-distributed Stochastic Neighbor Embedding is a non-linear dimensionality reduction method that converts similarities between data points to joint probabilities using the Student t-distribution in the low-dimensional space

3. UMAP
Uniform Manifold Approximation and Projection (UMAP) constructs a high-dimensional graph representation of the data then optimizes a low-dimensional graph to be as structurally similar as possible.

4. ICA
Independent Component Analysis (ICA) is a linear dimensionality reduction method that aims to separate a multivariate signal into additive subcomponents under the assumption that independent components are non-gaussian. Where PCA compresses the data, ICA separates the information

5. PaCMAP
Pairwise Controlled Manifold Approximation (PaCMAP) is a dimensionality reduction method that optimizes low-dimensional embeddings using three kinds of point pairs: neighbor pairs, mid-near pair, and further pairs

## Reinforcement Learning