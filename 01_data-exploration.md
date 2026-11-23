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
Output:
| index | customer_id | telecom_partner | gender | age | state | city | pincode | registration_event | num_dependents | estimated_salary |
|---:|---:|---|---|---:|---|---|---:|---|---:|---:|
| 0 | 15169 | Airtel | F | 26 | Himachal Pradesh | Delhi | 667173 | 2020-03-16 | 4 | 85979 |
| 1 | 149207 | Airtel | F | 74 | Uttarakhand | Hyderabad | 313997 | 2022-01-16 | 0 | 69445 |
| 2 | 148119 | Airtel | F | 54 | Jharkhand | Chennai | 549925 | 2022-01-11 | 2 | 75949 |
| 3 | 187288 | Reliance Jio | M | 29 | Bihar | Hyderabad | 230636 | 2022-07-26 | 3 | 34272 |
| 4 | 14016 | Vodafone | M | 45 | Nagaland | Bangalore | 188036 | 2020-03-11 | 4 | 34157 |

```python
telecom_usage.head()
```
Output:
| index | customer_id | calls_made | sms_sent | data_used | churn |
|---:|---:|---:|---:|---:|---:|
| 0 | 15169 | 75 | 21 | 4532 | 1 |
| 1 | 149207 | 35 | 38 | 723 | 1 |
| 2 | 148119 | 70 | 47 | 4688 | 1 |
| 3 | 187288 | 95 | 32 | 10241 | 1 |
| 4 | 14016 | 66 | 23 | 5246 | 1 |

### 1.3.2 Check the dataset dimensions
```python
print("Demographics shape:", telecom_demographics.shape)
print("Usage shape:", telecom_usage.shape)
```
Output:
```
Demographics shape: (6500, 10)
Usage shape: (6500, 5)
```

### 1.3.3 Inspect column names and data types
```python
telecom_demographics.info()
```
Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6500 entries, 0 to 6499
Data columns (total 10 columns):
 #   Column              Non-Null Count  Dtype 
---  ------              --------------  ----- 
 0   customer_id         6500 non-null   int64 
 1   telecom_partner     6500 non-null   object
 2   gender              6500 non-null   object
 3   age                 6500 non-null   int64 
 4   state               6500 non-null   object
 5   city                6500 non-null   object
 6   pincode             6500 non-null   int64 
 7   registration_event  6500 non-null   object
 8   num_dependents      6500 non-null   int64 
 9   estimated_salary    6500 non-null   int64 
dtypes: int64(5), object(5)
memory usage: 507.9+ KB
```

```python
telecom_usage.info()
```
Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6500 entries, 0 to 6499
Data columns (total 5 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   customer_id  6500 non-null   int64
 1   calls_made   6500 non-null   int64
 2   sms_sent     6500 non-null   int64
 3   data_used    6500 non-null   int64
 4   churn        6500 non-null   int64
dtypes: int64(5)
memory usage: 254.0 KB
```

### 1.3.4 Check missing values
```python
telecom_demographics.isnull().sum()
```
Output:
| Column             | Missing Values |
|--------------------|---------------:|
| index              | 0              |
| customer_id        | 0              |
| telecom_partner    | 0              |
| gender             | 0              |
| age                | 0              |
| state              | 0              |
| city               | 0              |
| pincode            | 0              |
| registration_event | 0              |
| num_dependents     | 0              |
| estimated_salary   | 0              |

```python
telecom_usage.isnull().sum()
```
Output:
| Column       | Missing Values |
|--------------|---------------:|
| index        | 0              |
| customer_id  | 0              |
| calls_made   | 0              |
| sms_sent     | 0              |
| data_used    | 0              |
| churn        | 0              |

### 1.3.5 Check duplicate records
```python
print("Demographics duplicates:", telecom_demographics.duplicated().sum())
print("Usage duplicates:", telecom_usage.duplicated().sum())
```
Output:
```
Demographics duplicates: 0
Usage duplicates: 0
```
