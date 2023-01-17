# notes on ml algo

## pca

1. can be used for facial recognition
2. It is a linear transformmation
3. most popular feature extraqction method
4. consisits of transformation from space of hign dim to another with more reduced dim
5. If data is highly corelated there is redundant information
6. so pca decreases amount of redundant information by decorrelating input vectors
7. input vectors with high dimentions & correlated can be represented in lower dimention space & decorrelated
8. PCA is powerful tool to compress data
9. Principal component analysis (PCA) is a technique used to reduce the dimensionality of a data set while retaining as much of the variation present in the data as possible. It does this by transforming the data to a new coordinate system such that the first axis (called the first principal component) explains the greatest amount of variance in the data, the second axis (second principal component) explains the second greatest amount of variance, and so on. These new axes are called principal components, and the data can be projected onto them to create a lower-dimensional representation of the data. PCA is often used as a tool in exploratory data analysis and for making data easy to visualize.

## Linear Discriminant analysis

1. Linear discriminant analysis (LDA) is a technique used to find a linear combination of features that separates different classes of a data set as well as possible. It is a supervised technique, meaning it requires labeled data. It is commonly used as a dimensionality reduction technique and a classifier. LDA is a method used for finding a linear combination of features that separates different classes of a data set as well as possible.

The goal of LDA is to project the original features onto a new feature space with a lower dimensionality while retaining as much of the class discrimination information as possible. LDA finds the linear combination of features that maximizes the ratio of the between-class variance to the within-class variance. In other words, it finds the feature space that separates the classes as much as possible while keeping the data points within each class as close together as possible.
2. LDA works better than pca when training data is well representative of data in system
3. If data isnt presentative enough pca works better

## Independent component analysis

## NMF

## AUC ROC CURVE

## k folds with cross validation


## classification

### Binary classifier

### F1

### Multiclass classifier

## clustering

### Hierarchical clustering

### k means

## Decision trees

### Information Gain / entropy

1. ID3
2. c4.5
3. C5
4. J48

### Gini Index

1. SPRINT
2. SLIQ

## Association rule

## Encoding

1. label encoder
2. One Hot Encoder

### Applications of ARL

1. Market Basket Analysis = big retailers
2. Medical diagnosis
3. Protein Sequence

### Association rule can be divided into 3 algorithms

1. Apriori => uses frequent datasets to generate association rules
It is designed to work on databases that contain transactions
It uses a breadth - first search and hash tree to calculate the itemset efficiently

Apriori Algorithm = uses frequent datasets to generate association rules
It is designed to work on databases that contain transactions

we use mlxtend to implement apriori
mlxtend.preprocessing import TransactionEncoder

It uses a breadth - first search and hash tree to calculate the itemset efficiently
2. EClat
3. F-G Growth

### Metrics of AR

1. Support
2. Confidence
3. Lift

## YOLO-v3

## Auto encoders

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
5. BERT
6. SPACY

### Propogation

1. Backward Propogation
2. forward propogation
3. Optimization Algorithm
4. weight & bias

## Linear Regression

1. Logistic Linear regression
2. Linear Regresssion (supervised learning)
targets predictive value
finds out relationship between variable and forcasting
3. multiple linear regression

## Anamoly detection (outlier detection)

### Supervised Anamoly detection (needs training of data)

1. KNN
2. Bayesian network

### Unsupervised Anamoly detection (Doesnt requires training data)

## simple imputer

## Bagging classifier

## Random forest

## Ensemble Learning

seeks better predictive performance by combining the predictions from multiple models

### Bagging

Bagging involves fitting many decision trees on different samples of the samedataset and averaging the predictions

1. Bagging classifier
2. Random forest
3. Extra trees
4. baggged decision trees

### Boosting

Boosting involves adding ensemble members sequentially that correct the predictions made by prior models and outputs a weighted
avg of the predictions

1. ada boost
2. XGBoost
3. Stochastic Gradient Boosting

### Stacking

Stacking involves fitting many different model types on the same data and using another model to learn how to best combine the predictions

### Voting

 Building multiple models (typically of differing types) and simple statistics (like calculating the mean) are used to combine predictions.

1. Hard voting
2. soft voting

### Blending

## Naive bayes

1. Gaussian naive bayes
2. naive bayes

## f1 score

## fitting

1. overfitting
2. underfitting

## svm

SVM - Supervised: Classification + Regression
Application: Face detection, image classification, text categorization

## tune params

1. regularization
2. Gamma
3. Kernal

## smoothing methods

1. weighted moving average
2. forcast evaluation
