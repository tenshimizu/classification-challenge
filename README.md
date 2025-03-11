# Spam Detector Project

## Overview

This project aims to build a model that can accurately classify emails as either spam or not spam (ham) using machine learning techniques. The notebook `spam_detector.ipynb` contains the code for data loading, preprocessing, model training, and evaluation. Two models, Logistic Regression and Random Forest Classifier, are implemented and compared.

## Data

The dataset used in this project is the Spambase dataset, which can be found at the UCI Machine Learning Library: [https://archive.ics.uci.edu/dataset/94/spambase](https://archive.ics.uci.edu/dataset/94/spambase).

A copy of the data is retrieved from: [https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv](https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv)

The dataset contains features representing word frequencies, character frequencies, and capital run lengths, along with a label indicating whether the email is spam (1) or not (0).

## Files

*   `spam_detector.ipynb`: Jupyter Notebook containing the code for the project.

## Libraries Used

*   pandas
*   scikit-learn (`sklearn`)
    *   `train_test_split`
    *   `StandardScaler`
    *   `LogisticRegression`
    *   `RandomForestClassifier`
    *   `accuracy_score`

## Workflow

1.  **Data Loading and Exploration:**
    *   The dataset is loaded using pandas.
    *   The first few rows of the DataFrame are displayed to verify successful import.

2.  **Data Preprocessing:**
    *   The data is split into features (X) and labels (y). The "spam" column is used as the label.
    *   The balance of the labels is checked using `value_counts()`.
    *   The data is split into training and testing sets using `train_test_split()`.

3.  **Feature Scaling:**
    *   `StandardScaler` is used to scale the features.
    *   The scaler is fit on the training data, and then used to transform both the training and testing data.

4.  **Model Training and Evaluation:**
    *   **Logistic Regression:**
        *   A Logistic Regression model is created with a `random_state` of 1 for reproducibility.
        *   The model is trained on the scaled training data.
        *   Predictions are made on the scaled testing data.
        *   The accuracy score is calculated to evaluate the model's performance.
    *   **Random Forest Classifier:**
        *   A Random Forest Classifier model is created with a `random_state` of 1.
        *   The model is trained on the scaled training data.
        *   Predictions are made on the scaled testing data.
        *   The accuracy score is calculated to evaluate the model's performance.

5.  **Model Comparison:**
    *   The accuracy scores of both models are compared to determine which performed better.
    *   Compares the results against any initial predictions made.

## How to Run the Code

1.  Make sure you have the required libraries installed (`pandas`, `scikit-learn`).  You can install them using pip:

    ```
    pip install pandas scikit-learn
    ```

2.  Open the `spam_detector.ipynb` file in Jupyter Notebook or JupyterLab.

3.  Run the cells sequentially to execute the code and train/evaluate the models.

## Results

The results section in the notebook displays the accuracy scores for both the Logistic Regression and Random Forest models, allowing for a direct comparison of their performance on the spam detection task. The Random Forest model is expected to outperform the Logistic Regression due to the complex relationships and high dimensionality of the data.

## Author
Erin Spencer-Priebe
