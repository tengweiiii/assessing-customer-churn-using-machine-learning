# 4.0 Model Training
The project requires us to train two classification models:
1. Logistic Regression
2. Random Forest Classifier
We will train both models using the same training dataset and evaluate them using the same testing dataset.

## 4.1 Train Logistic Regression
**What is Logistic Regression?**
Logistic Regression is a supervised machine learning algorithm commonly used for binary classification.

In our project, it predicts whether a customer will:
- Churn (`1`)
- Not churn (`0`)

```python
# Initialize Logistic Regression
logreg = LogisticRegression(random_state=42)

# Train the model
logreg.fit(X_train, y_train)
```
**Why `random_state=42`?**

It helps make the model's results reproducible.

### 4.1.1 Make predictions
After training, use the testing dataset to predict customer churn.
```python
# Make predictions using Logistic Regression
logreg_pred = logreg.predict(X_test)
```
The predictions will contain binary values:
- `0`: Customer predicted not to churn.
- `1`: Customer predicted to churn.

Inspect the first 10 predictions:
```python
logreg_pred[:10]
```
Output:
```
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```
