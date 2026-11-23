# 1.0 Data Exploration
## Understand the project objective
Our goal is to predict whether a telecom customer will leave their current telecom provider

We will compare two classification algorithms:
1. **Logistic Regression**: Predicts the probability of a customer churning
2. **Random Forest Classifier**: Uses multiple decision trees to make predictions.

Our final objective is to determine which model achieves the higher accuracy score.

Understanding the datasets
| Dataset | Purpose | Records |
|---|---|---|
| `telecom_demographics.csv` | Customer demographic information | To investigate |
| `telecom_usage.csv` | Customer usage patterns and churn status | To investigate |

We will merge them using `customer_id`.

Our final DataFrame will be called `churn_df`.

## 1.1 Import libraries
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## 1.2 Load the datasets
```python
# Load the datasets
telecom_demographics = pd.read_csv('telecom_demographics.csv')
telecom_usage = pd.read_csv('telecom_usage.csv')
```

## 1.3 Inspect the datasets
### 1.3.1 Display the first five rows
```python
telecom_demographics.head()
```

```python
telecom_usage.head()
```

### 1.3.2 Check the dataset dimensions
```python
print("Demographics shape:", telecom_demographics.shape)
print("Usage shape:", telecom_usage.shape)
```

### 1.3.3 Inspect column names and data types
```python
telecom_demographics.info()
```

```python
telecom_usage.info()
```

### 1.3.4 Check missing values
```python
telecom_demographics.isnull().sum()
```

```python
telecom_usage.isnull().sum()
```

### 1.3.5 Check duplicate records
```python
print("Demographics duplicates:", telecom_demographics.duplicated().sum())
print("Usage duplicates:", telecom_usage.duplicated().sum())
```
