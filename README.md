# Customer Subscription Prediction Model

## Project Overview
This project aims to create a classification model to predict customer subscription to term deposits. The model employs multiple algorithms including K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree, Random Forest, and Boosting techniques. The goal is to accurately identify potential customers likely to subscribe to term deposits.

## Table of Contents
1. [Problem Identification](#problem-identification)
2. [Data Loading](#data-loading)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Implementation](#model-implementation)
5. [Evaluation](#evaluation)
6. [Conclusion](#conclusion)
7. [Installation](#installation)

## Problem Identification
The objective is to predict whether a customer will subscribe to a term deposit based on various features. This classification task helps financial institutions target their marketing efforts more effectively.

## Data Loading
The dataset includes customer information and features relevant to subscription prediction. The data used is Bank Marketing from UC Irvine that can be accessed [in this link](https://archive.ics.uci.edu/dataset/222/bank+marketing)

Example:
```python
import pandas as pd

# Load data from a CSV file
data = pd.read_csv('term_deposit_data.csv')
```

## Data Preprocessing
The data preprocessing consists of processes as follows.
1. Feature Selection
   As the target is a categorical value (of yes or no), feature selection is done using ANOVA correlation for numerical - categorical and chi-squared test for categorical - categorical.
3. Missing Value Handling
   As the missing value is inputed as `unknown` value, we do data imputation to the `unknown` values by creating a variable with binary value for the data with unknown data (1) and non unknown data (0).
4. Outlier Handling
   Outlier handling done using Winsorizer to the `duration` feature.
5. Feature Transformation
   The feature transformation consist of encoding for categorical data (using Binary Encoding and OneHot Encoding) and Scaling for numerical data.
6. Data Balancing
   Data is balanced using SMOTE.

## Model Implementation
The project implements multiple classification algorithms to predict customer subscription. The model used consist of :
1. Random Forest
2. SVM
3. KNN
4. Decision Tree
5. Ada Boosting
6. Gradient Boosting

## Evaluation
The models are evaluated using F1-score that harmonizes both precision and recall. Cross-validation is used to ensure robust performance measurement. Below are the outcome:

| Metrics | Random Forest | SVM | KNN | Decision Tree | Ada Boost | Gradient Boost |
| --- | --- | --- | --- | --- | --- | --- |
| train - f1 score | 0.993976	| 0.487273 | 0.612164 |	0.477922 |	0.421308 |	0.529471 |
| cross val f1 score mean | 0.385535 |	0.474722 |	0.402831 |	0.444867 |	0.393308 |	0.460552 |
| test - f1 score | 0.405303 |	0.464832 |	0.380165 |	0.381992 |	0.413655 |	0.448445 |

## Conclusion
This project predicts term deposit subscriptions using classification models like Random Forest, SVM, KNN, Decision Tree, and Boosting. Key features include duration, previous, balance, output, contact, housing, and loan. SVM emerged as the best model after tuning, with married, loan-free individuals aged 30-40 being likely subscribers. Optimal call duration is around 9 minutes. Further modeling should refine feature selection.

## Installation

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/vaniaalya14/bank-term-deposit-classification.git
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
