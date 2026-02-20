# KNN Classifier on Iris Dataset

This notebook demonstrates the implementation of a K-Nearest Neighbors (KNN) classifier using a custom Iris dataset (`Iris.csv`).

## Project Overview

The goal of this project is to:
1. Load and preprocess the `Iris.csv` dataset.
2. Train a KNN classifier.
3. Evaluate the model's performance.
4. Visualize the effect of different `k` values on accuracy.
5. Visualize the decision boundary of the KNN classifier using two features.

## Dataset

The dataset used is `Iris.csv`, which contains information about iris flowers including sepal length, sepal width, petal length, petal width, and species.

## Steps Performed

1.  **Load Libraries and Dataset**: Necessary libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`) are imported, and the `Iris.csv` dataset is loaded into a pandas DataFrame.
2.  **Data Preprocessing**: 
    *   The 'Id' column is dropped from the DataFrame.
    *   The 'Species' column (target variable) is encoded using `LabelEncoder` to convert categorical labels into numerical representations.
3.  **Data Splitting**: The features (X) and target (y) are defined. The dataset is then split into training and testing sets (70% training, 30% testing).
4.  **Feature Scaling**: Features are scaled using `StandardScaler` to normalize the data, which is crucial for distance-based algorithms like KNN.
5.  **Model Training and Evaluation**: A KNN classifier with `n_neighbors=5` is trained on the scaled training data. The model's performance is evaluated using `accuracy_score`, `confusion_matrix`, and `classification_report` on the test set.
6.  **K-Value Optimization**: The accuracy of the KNN model is evaluated for `k` values ranging from 1 to 20. A plot of 'K vs Accuracy' is generated to help visualize the impact of `k` on model performance.
7.  **Decision Boundary Visualization**: To visually understand the classifier's decision regions, a KNN model is trained using only two features (`SepalLengthCm` and `SepalWidthCm`). A decision boundary plot is generated, showing the classification regions and the training data points.

## Results

*   The KNN model achieved high accuracy on the Iris dataset after preprocessing and scaling.
*   The 'K vs Accuracy' plot provides insights into how different `k` values affect the model's predictive power.
*   The decision boundary visualization illustrates the classifier's separation of the different iris species based on the selected features.
