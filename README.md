# Breast-Cancer-Prediction-Model

#Overview
This repository contains a breast cancer prediction model developed using logistic regression. The model aims to predict the likelihood of a breast cancer diagnosis based on input features. This project utilizes a dataset containing various features.

#Introduction
Breast cancer is a significant health issue worldwide, and early detection is crucial for effective treatment. This model leverages logistic regression, a powerful statistical method, to classify whether a tumor is benign or malignant based on a set of features derived from breast cancer biopsy data.

#Dataset
The dataset used in this project is the Breast Cancer Data Set referred from the Kaggle site.

-Number of Instances: 569
-Number of Attributes: 32 
-Attribute Information:
-Mean radius
-Mean texture
-Mean perimeter
-Mean area
-Mean smoothness
... (and other relevant features)
-Target Variable: Diagnosis (M = malignant, B = benign)

#Model
The logistic regression model is used to estimate the probability that a given instance (tumor) is malignant. Logistic regression is well-suited for binary classification tasks.

#Model Training
-The dataset is split into training and testing sets.
-Feature scaling is performed to standardize the features.
-The logistic regression algorithm is applied to the training data.

#Model Evaluation
The model's performance is evaluated using metrics such as accuracy and precision.

#Evaluation
The model's performance is assessed using the following metrics:

-Accuracy: The proportion of true results (both true positives and true negatives) among the total number of cases examined.
-Precision: The proportion of true positive results among the total predicted positives.

#Results
The logistic regression model achieves an accuracy of 97.9% on the test dataset. The results demonstrate the model's effectiveness in predicting breast cancer diagnoses.
