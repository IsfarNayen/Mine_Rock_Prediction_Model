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

**Final Model Selection**
The Logistic regression was selected as the final model due to its superior performance on the test dataset.

<h2>Performance Results:</h2>

|  Model   | Training Accuracy |  Test Accuracy |
|----------|-------------------|----------------|
|Logistic Regression| 83.422%  |       76.19%   |

<h2> My Contributions and Enhancements </h2>

Beyond the basic tutorial implementation, I've made several key improvements to the project:

**Enhanced Input Processing**
Added flexible input handling that supports both NumPy and Pandas arrays:

```python
# Example input data
# 0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032

# Splitting the input data based on comma
input_data = list(map(float, input().split(',')))

# Creating numpy array, because model prediction accepts only numpy or pandas array
array = np.asarray(input_data) 

# Added Pandas DataFrame support for prediction (enhancement beyond the tutorial)
pandas_array = pd.DataFrame([input_data])
```