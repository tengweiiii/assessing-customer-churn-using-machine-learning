# 3.0 Data Preparation for Modelling
## Train-Test Split
We need to divide our dataset into:
- Training set (80%): Used to train the machine learning models.
- Testing set (20%): Used to evaluate how well the models perform on unseen data.

## 3.1 Prepare the training and testing data
Before splitting, let's address the data leakage issue mentioned earlier.

We previously fitted OneHotEncoder and StandardScaler using all 6,500 records. For a correct evaluation, the encoder and scaler should only learn from the training data.

We will split the original features first, then fit preprocessing on the training set.

```python
# Define input features
X = churn_df[categorical_features + numerical_features]

# Define target variable
y = churn_df['churn']
```

## 3.2 Split the dataset
Use train_test_split() from Scikit-learn.
```python
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.2,
  random_state=42,
  stratify=y
)
```

**Understanding the parameters**
| Parameter | Purpose |
|---|---|
| `X` | Input features |
| `y` | Target variable |
| `test_size=0.2` | Allocate 20% for testing |
| `random_state=42` | Ensure reproducibility |
| `stratify=y` | Preserve the churn proportion in both sets |

We use `stratify=y` because our target variable is imbalanced (approximately 80% non-churn and 20% churn).

## 3.3 Apply preprocessing to training and testing sets
Now we fit the encoder and scaler using training data only.
```python
# Initialize OneHotEncoder and StandardScaler
encoder = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()

# Fit and transform training features
encoded_train = encoder.fit_transform(
  X_train_raw[categorical_features]
).toarray()

scaled_train = scaler.fit_transform(
  X_train_raw[numerical_features]
)

# Transform testing features
encoded_test = encoder.transform(
  X_test_raw[categorical_features]
).toarray()

scaled_test = scaler.transform(
  X_test_raw[numerical_features]
)

# Combine encoded and scaled features
import numpy as np

X_train = np.concatenate(
  [scaled_train, encoded_train],
  axis=1
)

X_test = np.concatenate(
  [scaled_test, encoded_test],
  axis=1
)
```
Notice that we use `fit_transform()` for training data but only transform() for testing data.

This prevents information from the testing set from influencing the preprocessing.

## 3.4 Check the training and testing dimensions
```python
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
```
The number of feature columns should remain 46 if the training data contains the same categories as your earlier encoding.

Output:
```
X_train: (5200, 46)
X_test: (1300, 46)
y_train: (5200,)
y_test: (1300,)
```

## Review train-test split
| Variable | Shape | Percentage |
|---|---|---:|
| `X_train` | (5200, 46) | 80% |
| `X_test` | (1300, 46) | 20% |
| `y_train` | (5200,) | 80% |
| `y_test` | (1300,) | 20% |

A few things to confirm:
- Training and testing data have the same 46 features.
- The training set contains 5,200 customers.
- The testing set contains 1,300 customers.
- `random_state=42` ensures reproducibility.
- `stratify=y` preserves the churn distribution.
  
