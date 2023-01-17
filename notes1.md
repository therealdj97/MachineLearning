# ML Algorithms

## Supervised Learning

### Tree Based

1. Decision Tree : Decision Tree models make decision rules on the features to produce predictions. It can be used for classification or regression
    1. Information Gain / Entropy
        1. ID3
        2. c4.5
        3. C5
        4. J48

    2. Gini Index
        1. SPRINT
        2. SLIQ

2. Random Forest :
An ensemble learning method that combines the output of multiple decision trees

3. Gradient Boosting Regression :
Gradient Boosting Regression employs boosting to make predictive models from an ensemble of weak predictive learners

4. XGBoost :
Gradient Boosting algorithm that is eOcient & flexible. Can be used for both classification and regression tasks

5. LightGBM Regression :
A gradient boosting framework that is designed to be more effcient than other implementations

### Linear Models

1. Linear Regression :
A simple algorithm that models a linear relationship between inputs and a continuous numerical output variable |(multiple linear regression) targets predictive value
finds out relationship between variable and forcasting

2. Logistic Regression :
A simple algorithm that models a linear relationship between inputs and a categorical output (1 or 0)

3. Ridge Regression :
Part of the regression family — it penalizes features that have low predictive outcomes by shrinking their coefficients closer to zero. Can be used for classification or regression

4. Lasso Regression :
Part of the regression family — it penalizes features that have low predictive outcomes by shrinking their coeOcients to zero. Can be used for classification or regression

### Regression Only

1. Linear Regression :
Linear Regression models a linear relationship between input variables and a continuous numerical output variable. The default loss function is the mean square error (MSE)

2. Polynomial regression :
Polynomial Regression models nonlinear relationships between the dependent, and independent variable as the n-th degree polynomial.

3. Support vector Regression :
Support Vector Regression (SVR) uses the same principle as SVMs but optimiRes the cost function to fit the most straight line (or plane) through the data points. With the kernel trick it can efficiently perform a non-linear regression by implicitly mapping their inputs into high-dimensional feature spaces.

4. Gausian Process Regression :
Gaussian Process Regression (GPR) uses a Bayesian approach that infers a probability distribution over the possible functions that fit the data. The Gaussian process is a prior that is specified as a multivariate Gaussian distribution

5. Robust Regression :
Robust Regression is an alternative to least squares regression when data is contaminated with outliers. The term “robust” refers to the statistical capability to provide useful information even in the face of outliers.

### Both regression / classification models

1. Decision Trees :
Decision Tree models learn on the data by making decision rules on the variables to separate the classes in a flowchart like a tree data structure. They can be used for both regression and classification.

2. Random Forest :
Random Forest classification models learn using an ensemble of decision trees. The output of the random forest is based on a majority vote of the different decision trees

3. Gradient Boosting :
An ensemble learning method where weak predictive learners are combined to improve accuracy. Popular techniques include XGBoost, LightGBM and more.

4. Ridge Regression :
Ridge Regression penaliRes variables with low predictive outcomes by shrinking their coefficients towards Rero. It can be used for classification and regression.

5. Lasso Regression :
Lasso Regression penaliRes features that have low predictive outcomes
by shrinking their coefficients to Rero. It can be used for classification
and regression.

6. AdaBoost :Adaptive Boosting uses an ensemble of weak learners that is combined into a weighted sum that represents the final output of the boosted classifier.
    1. Advantages
        1. Explainable & Interpretable
        2. Less need for tweaking parameters
        3. Usually outperforms Random Forest

    2. Disadvantages
        1. Less prone to overfitting as the input variables are not jointly optimized
        2. ensitive to noisy data and outliers.

### Classification

1. SVM : (Classification + Regression)
In its simplest form, support vector machine is a linear classifier. But with the kernel trick, it can efficiently perform a non-linear classification by implicitly mapping their inputs into high-dimensional feature spaces. This makes SVM one of the best prediction methods. | Application: Face detection, image classification, text categorization

2. Nearest Neighbors :
Nearest Neighbors predicts the label based on a predefined number of samples closest in distance to the new point.

3. Logistic Regresssion (& its extensions) :
The logistic regression models a linear relationship between input variables and the response variable. It models the output as binary values (0 or 1) rather than numeric values.

4. Linear Descriminent Analysis :
The linear decision boundary maximiRes the separability between the classes by finding a linear combination of features

 Linear discriminant analysis (LDA) is a technique used to find a linear combination of features that separates different classes of a data set as well as possible. It is a supervised technique, meaning it requires labeled data. It is commonly used as a dimensionality reduction technique and a classifier. LDA is a method used for finding a linear combination of features that separates different classes of a data set as well as possible.

The goal of LDA is to project the original features onto a new feature space with a lower dimensionality while retaining as much of the class discrimination information as possible. LDA finds the linear combination of features that maximizes the ratio of the between-class variance to the within-class variance. In other words, it finds the feature space that separates the classes as much as possible while keeping the data points within each class as close together as possible.
2. LDA works better than pca when training data is well representative of data in system
3. If data isnt presentative enough pca works better

## Unsupervised Learning

### Association

1. Apriori :
Rule based approach that identifies the most frequent itemset in a given dataset where prior knowledge of frequent itemset properties is used
The Apriori algorithm uses the join and prune step iteratively to identify the most frequent itemset in the given dataset. Prior knowledge (apriori) of frequent itemset properties is used in the process.
uses frequent datasets to generate association rules
It is designed to work on databases that contain transactions
It uses a breadth - first search and hash tree to calculate the itemset efficiently
Apriori Algorithm = uses frequent datasets to generate association rules
It is designed to work on databases that contain transactions
we use mlxtend to implement apriori.
`mlxtend.preprocessing import TransactionEncoder`

2. FP_Growth algorithm :
Frequent Pattern growth (FP-growth) is an improvement on the Apriori algorithm for finding frequent itemsets. It generates a conditional FP-Tree for every item in the data

3. FP-Max Algorithm :
A variant of Frequent pattern growth that is focused on finding maximal itemsets

4. Eclat :
Equivalence Class Clustering and Bottom-Up Lattice Traversal (Eclat) applies a Depth First Search of a graph procedure. This is a more efficient and scalable version of the
Apriori algorithm.

5. Hypergeometric Networks :
HNet learns the Association from datasets with mixed data types (discrete and continuous variables) and with unknown functions. Associations are statistically tested using the hypergeometric distribution for finding frequent itemset.

#### Applications for Association Rule learning

1. Market Basket Analysis = big retailers
2. Medical diagnosis
3. Protein Sequence

### Clustering

1. K-Means :
KMeans is the most widely used clustering approach—it determines K clusters based on euclidean distances
Most common clustering approach which assumes that the closer data points are to each other, the more similar they are. It determines K clusters based on Euclidean distances.

2. Hierarchical clustering :
A bottomlup approach where each data point is treated as its own cluster—and then the closest two clusters are merged together iteratively

3. Gausian Mixture :
A bottomlup approach where each data point is treated as its own cluster—and then the closest two clusters are merged together iteratively

4. DBSCAN :
Density-Based Spatial Clustering of Applications with Noise can handle non-linear cluster structures, purely based on density. It can differentiate and separate regions with varying degrees of density, thereby creating clusters.

5. HDBSCAN :
Family of the density-based algorithms and has roughly two steps: finding the core distance of each point, and expands clusters from them. It extends DBSCAN by converting it into a hierarchical clustering algorithm

6. Agglomerative Hierarchical Clustering :
Uses hierarchical clustering to determine the distance between samples based on the metric, and pairs are merged into clusters using the linkage type.

7. OPTICS :
Family of the density-based algorithms where it finds core sample of high density and expands clusters from them. It operates with a core distance (ɛ) and reachability distance

8. Gausian Mixture Models :
Gaussian Mixture Models (GMM) leverages probabilistic models to detect clusters using a mixture of normal (gaussian) distributions.

### Dimensionality Reduction

1. PCA :
Principal Component Analysis (PCA) is a feature extraction approach that uses a linear function to reduce dimensionality in datasets by minimizing information loss.
    1. can be used for facial recognition
    2. It is a linear transformmation
    3. most popular feature extraqction method
    4. consisits of transformation from space of hign dim to another with more reduced dim
    5. If data is highly corelated there is redundant information
    6. so pca decreases amount of redundant information by decorrelating input vectors
    7. input vectors with high dimentions & correlated can be represented in lower dimention space & decorrelated
    8. PCA is powerful tool to compress data
    9. Principal component analysis (PCA) is a technique used to reduce the dimensionality of a data set while retaining as much of the variation present in the data as possible. It does this by transforming the data to a new coordinate system such that the first axis (called the first principal component) explains the greatest amount of variance in the data, the second axis (second principal component) explains the second greatest amount of variance, and so on. These new axes are called principal components, and the data can be projected onto them to create a lower-dimensional representation of the data. PCA is often used as a tool in exploratory data analysis and for making data easy to visualize.

2. t-SNE :
t-distributed Stochastic Neighbor Embedding is a non-linear dimensionality reduction method that converts similarities between data points to joint probabilities using the Student t-distribution in the low-dimensional space

3. UMAP :
Uniform Manifold Approximation and Projection (UMAP) constructs a high-dimensional graph representation of the data then optimizes a low-dimensional graph to be as structurally similar as possible.

4. ICA :
Independent Component Analysis (ICA) is a linear dimensionality reduction method that aims to separate a multivariate signal into additive subcomponents under the assumption that independent components are non-gaussian. Where PCA compresses the data, ICA separates the information

5. PaCMAP :
Pairwise Controlled Manifold Approximation (PaCMAP) is a dimensionality reduction method that optimizes low-dimensional embeddings using three kinds of point pairs: neighbor pairs, mid-near pair, and further pairs

## Reinforcement Learning

## Decision trees


## Ensemble Learning

seeks better predictive performance by combining the predictions from multiple Macine learning  models

### Boosting

Boosting involves adding ensemble members sequentially that correct the predictions made by prior models and outputs a weighted
avg of the predictions

1. ada boost
2. XGBoost
3. Stochastic Gradient Boosting

### Bayesian model averaging

### Bayesian model combination

### Bucket of models

### Stacking

Stacking involves fitting many different model types on the same data and using another model to learn how to best combine the predictions

### Bagging

Bagging involves fitting many decision trees on different samples of the samedataset and averaging the predictions

1. Bagging classifier
2. Random forest
3. Extra trees
4. baggged decision trees

### Voting

 Building multiple models (typically of differing types) and simple statistics (like calculating the mean) are used to combine predictions.

1. Hard voting
2. soft voting

### Blending

## Nural Networks

1. Convolution nural networks
2. Recurrent nural networks
3. Artificial Nural Networks
4. Layer Nural network

## NLP

1. Algorithmia provides a free API endpoint for many algorithms
2. NLTK (Natural Language toolkit) a Python library that provides modules for processing text, classifying, tokenizing, tagging, parsing, and more
3. Stanford NLP :  a suite of NLP tools that provide part-of-speech tagging, the named entity recognizer, coreference resolution system, sentiment analysis, and more.
4. Google SyntaxNet a neural-network framework for analyzing and understanding the grammatical structure of sentences.
5. BERT : Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google
6. SPACY : free open-source library for Natural Language Processing in Python. It features NER, POS tagging, dependency parsing, word vectors and more

## HyperParameters / Tune Parameters

The parameters which can affect your output

1. regularization
2. Gamma
3. Kernal
