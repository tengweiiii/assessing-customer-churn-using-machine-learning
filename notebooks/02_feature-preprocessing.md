# 2.0 Feature Preprocessing

The project asks us to:
1. Convert categorical features into numerical representations.
2. Perform feature scaling.
3. Define `features_scaled`.
4. Define the `target` variable.
Before writing the code, let's understand which columns we should use.

## 2.1 Understand the features
Our merged dataset contains 14 columns.
| Column | Data Type | Treatment |
|---|---|---|
| `customer_id` | Numerical | Exclude (identifier) |
| `telecom_partner` | Categorical | One-Hot Encoding |
| `gender` | Categorical | One-Hot Encoding |
| `age` | Numerical | StandardScaler |
| `state` | Categorical | One-Hot Encoding |
| `city` | Categorical | One-Hot Encoding |
| `pincode` | Numerical | Exclude (location identifier) |
| `registration_event` | Date | Exclude for this baseline model |
| `num_dependents` | Numerical | StandardScaler |
| `estimated_salary` | Numerical | StandardScaler |
| `calls_made` | Numerical | StandardScaler |
| `sms_sent` | Numerical | StandardScaler |
| `data_used` | Numerical | StandardScaler |
| `churn` | Target | Target variable |

**Why exclude certain columns?**

- `customer_id`: Only identifies customers. It does not represent customer behaviour.
- `pincode`: A location code, not a continuous numerical measurement.
- `registration_event`: A date stored as text. We would need to engineer meaningful date features before using it.
This gives us four categorical features and five numerical features.

## 2.2 Separate categorical and numerical features
```python
# Define categorical features
categorical_features = [
  'telecom_partner',
  'gender',
  'state',
  'city'
]

# Define the numerical features
numerical_features = [
  'age',
  'num_dependents',
  'estimated_salary',
  'calls_made',
  'sms_sent',
  'data_used'
]
```
Notice that we have six numerical features, not five. The target is excluded from the feature list.

## 2.3 Apply One-Hot Encoding
Machine learning algorithms require numerical inputs.

However, categorical features such as telecom_partner contain text values:
- Airtel
- Reliance Jio
- Vodafone
- BSNL
  
We can use `OneHotEncoder` to convert these categories into numerical columns.

For example:
| Telecom Partner | Airtel | Reliance Jio | Vodafone |
|---|---:|---:|---:|
| Airtel | 1 | 0 | 0 |
| Reliance Jio | 0 | 1 | 0 |
| Vodafone | 0 | 0 | 1 |

```python
# Initialize OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')

# Encode categorical features
encoded_features = encoder.fit_transform(
  churn_df[categorical_features]
).toarray()
```

**Why `handle_unknown='ignore'`?**

It prevents errors if the encoder encounters a category that it has not seen before.
For example, if a new telecom partner appears in future data, the encoder can handle it without raising an error.

## 2.4 Apply StandardScaler
StandardScaler transforms numerical values so that they have approximately:
- Mean = 0
- Standard deviation = 1
This is useful for Logistic Regression because features with very different numerical ranges can otherwise affect model training.

```python
# Initialize StandardScaler
scaler = StandardScaler()

# Scale numerical features
scaled_features = scaler.fit_transform(
  churn_df[numerical_features]
)
```

## 2.5 Combine encoded and scaled features
```python
import numpy as np

# Combine categorical and numerical features
features_scaled = np.concatenate(
  [scaled_features, encoded_features],
  axis=1
)

# Inspect dimensions
features_scaled.shape
```
The number of rows should remain 6,500.

The number of columns will depend on the unique categories found in the four categorical features.

Output:
```
(6500, 46)
```

**Important improvement for a production-quality workflow**: We are demonstrating the requested preprocessing steps here, but fitting the encoder and scaler on the entire dataset before train-test splitting introduces data leakage. We will correct this by fitting preprocessing on training data only when we build the final modelling pipeline.

## 2.6 Define the target variable
Our target variable is `churn`.
```python
target = churn_df['churn']
```

### 2.6.1 Check the dimensions
```python
print("Features shape:", features_scaled.shape)
print("Target shape:", target.shape)
```
Output:
```
Features shape: (6500, 46)
Target shape: (6500,)
```

## 2.7 Review preprocessing results
### 2.7.1 Feature dimensions
| Variable | Shape | Explanation |
|---|---|---|
| `encoded_features` | Depends on categories | One-hot encoded categorical features |
| `scaled_features` | (6500, 6) | Six scaled numerical features |
| `features_scaled` | (6500, 46) | Combined numerical and categorical features |
| `target` | (6500,) | Churn labels |

Your final feature matrix contains:
- 6 numerical features.
- 40 encoded categorical features.
- 46 total features.

The 40 categorical columns come from the unique categories across `telecom_partner`, `gender`, `state`, and `city`.

### 2.7.2 Why are there 46 features?
One-Hot Encoding creates a separate column for each unique category.

For example, if your categorical columns contain:
| Feature | Number of Unique Categories |
|---|---:|
| telecom_partner | 4 |
| gender | 2 |
| state | 28 |
| city | 6 |
| **Total** | **40** |

Therefore:

$$
6 + 40 = 46
$$

This explains your output.

You can verify the category counts using:
```python
churn_df[categorical_features].nunique()
```
The total should be 40.

Output:
| Feature | Number of Unique Categories |
|---|---:|
| index | 0 |
| telecom_partner | 4 |
| gender | 2 |
| state | 28 |
| city | 6 |
