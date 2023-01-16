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
* 1. Explainable & Interpretable
* 2. Less need for tweaking parameters
* 3. Usually outperforms Random Forest

* Disadvantages
* 1. Less prone to overfitting as the input variables are not jointly optimiRed@
* 2. ensitive to noisy data and outliers.

### Classification

1. SVM
2. Nearest Neighbors
3. Logistic Regresssion (& its extensions)
4. Linear Descriminent Analysis

## Unsupervised Learning

### Association

1. Apriori
2. FP_Growth algorithm
3. FP-Max Algorithm
4. Eclat
5. Hypergeometric Networks

### Clustering

1. K-Means
2. Hierarchical clustering
3. Gausian Mixture
4. DBSCAN
5. HDBSCAN
6. Agglomerative Hierarchical Clustering
7. OPTICS
8. Gausian Mixture Models

### Dimensionality Reduction

1. PCA
2. t-SNE
3. UMAP
4. ICA
5. PaCMAP

## Reinforcement Learning
