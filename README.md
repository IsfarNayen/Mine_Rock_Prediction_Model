# Mine_Rock_Prediction_Model

<h2>Project Overview:</h2>
This project uses the Sonar dataset to build a machine learning model that can classify objects as either rocks or mines based on sonar readings. The model could be valuable in underwater object detection and maritime safety applications.

<h2>Dataset Information:</h2>

The dataset used in this project is the Sonar dataset, which contains:

208 data points with 60 attributes each
Each attribute represents the energy within a particular frequency band
Each data point is labeled as either a rock (R) or a mine (M)

<h2>Model Development Process</h2>

**Data Preprocessing**

Data loading and inspection
Feature and target separation
Training and testing data split (90/10 ratio)

**Model Selection**

The project evaluates and compares multiple classification algorithms:

- Logistic Regression

**Model Training**

Each model is trained using the training dataset with the default hyperparameters.

**Model Evaluation**

Models are evaluated using:
Accuracy score on both training and test datasets
Comparison of performance across all models

Final Model Selection
The Logistic regression was selected as the final model due to its superior performance on the test dataset.