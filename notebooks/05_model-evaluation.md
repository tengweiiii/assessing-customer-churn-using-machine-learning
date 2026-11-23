# 5.0 Model Evaluation
Now that both models are trained, we need to compare their performance.

The project specifically asks us to compare their accuracy scores.

## 5.1 Calculate Logistic Regression accuracy
```python
logreg_accuracy = accuracy_score(y_test, logreg_pred)
print("Logistic Regression Accuracy:", logreg_accuracy)
```
Output:
```
Logistic Regression Accuracy: 0.7992307692307692
```

## 5.2 Calculate Random Forest accuracy
```python
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)
```
Output:
```
Random Forest Accuracy: 0.7984615384615384
```

Accuracy measures the proportion of correctly predicted observations.

$$
Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}
$$

For example, an accuracy of 0.85 means the model correctly predicted 85% of the testing observations.

## 5.3 Compare model accuracy
```python
print("Logistic Regression:", logreg_accuracy)
print("Random Forest:", rf_accuracy)
```
Output:
```
Logistic Regression: 0.7992307692307692
Random Forest: 0.7984615384615384
```

## 5.4 Determine the model with higher accuracy
The project requires us to assign either "LogisticRegression" or "RandomForest" to `higher_accuracy`.
```python
if logreg_accuracy > rf_accuracy:
    higher_accuracy = "LogisticRegression"
else:
    higher_accuracy = "RandomForest"

print("Higher Accuracy Model:", higher_accuracy)
```
Output:
```
Higher Accuracy Model: LogisticRegression
```

**Note**: If both models achieve exactly the same accuracy, this code selects Random Forest.
