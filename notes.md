# doubts on ml algo

## Propogation

1. Backward Propogation
2. forward propogation
3. Optimization Algorithm
4. weight & bias

## Binary classifier

it uses confusion matrix & supervised learning algorithm for that

    Support Vector Machines
    Naive Bayes
    Nearest Neighbor
    Decision Trees
    Logistic Regression
    Neural Networks

    few binary classification applications, where the 0 and 1 columns are two possible classes for each observation:

    1.    Application -> Observation -> 0 1
    2.    Medical Diagnosis -> Patient -> Healthy -> Diseased
    3.    Email Analysis -> Email -> Not Spam -> Spam
    4.    Financial Data Analysis -> Transaction -> Not Fraud -> Fraud
    5.    Marketing -> Website visitor -> Won't Buy -> Will Buy
    6.    Image Classification -> Image -> Hotdog -> Not Hotdog

    https://towardsdatascience.com/top-10-binary-classification-algorithms-a-beginners-guide-feeacbd7a3e2

## Multiclass classifier

In the case of identification of different types of fruits, “Shape”, “Color”, “Radius” can be featured, and “Apple”, “Orange”, “Banana” can be different class labels.

supervised learning technique
we can check accuracy

Approach –  

1. Load dataset from the source.
2. Split the dataset into “training” and “test” data.
3. Train Decision tree, SVM, and KNN classifiers on the training data.
4. Use the above classifiers to predict labels for the test data.
5. Measure accuracy and visualize classification.

## simple imputer

https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html

## smoothing methods

used to reduce effect of Variation/Veriance on dataset
used in forcasting techniqueqs

1. weighted moving average
2. forcast evaluation
3. 