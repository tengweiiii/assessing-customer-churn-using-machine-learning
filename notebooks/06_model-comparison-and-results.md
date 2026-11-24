# 6.0 Model Comparison and Results
## 6.1 Compare accuracy scores
| Model | Accuracy |
|---|---:|
| Logistic Regression | 0.79923 |
| Random Forest | 0.79846 |

Logistic Regression performs slightly better than Random Forest.

However, the difference is extremely small, so we should not conclude that Logistic Regression is significantly better overall.

## 6.2 Interpret the results
A few observations:
1. Both models achieve approximately 80% accuracy.
2. Logistic Regression achieves a slightly higher accuracy score.
3. Random Forest performs almost identically to Logistic Regression.
4. The results suggest that Logistic Regression is competitive for this dataset.

**Important observation:**
Remember that approximately 80% of customers in our original dataset did not churn.
Our models achieve approximately 80% accuracy, which means we should investigate whether they are actually identifying customers who churn or simply predicting the majority class.

## 6.3 Examine the confusion matrices
| Term | Meaning |
|---|---|
| True Negative (TN) | Correctly predicted not churned |
| False Positive (FP) | Predicted churned, but actually not churned |
| False Negative (FN) | Predicted not churned, but actually churned |
| True Positive (TP) | Correctly predicted churned |

```python
from sklearn.metrics import confusion_matrix

# Logistic Regression confusion matrix
print("Logistic Regression: ")
print(confusion_matrix(y_test, logreg_pred))

# Random Forest confusion matrix
print("\nRandom Forest:")
print(confusion_matrix(y_test, rf_pred))
```

Output:
```
Logistic Regression: 
[[1039    0]
 [ 261    0]]

Random Forest:
[[1038    1]
 [ 261    0]]
```

## 6.4 Examine classification reports
The classification report provides additional evaluation metrics:
- Precision
- Recall
- F1-score
- Support
  
``` python
from sklearn.metrics import classification_report

# Logistic Regression classification report
print("Logistic Regression:")
print(classification_report(y_test, logreg_pred))

# Random Forest classification report
print("\nRandom Forest:")
print(classification_report(y_test, rf_pred))
```
Pay particular attention to the recall score for class 1.

Recall tells us how many actual churned customers were successfully identified by the model.

For telecom companies, identifying customers who are likely to leave is important because they can take action to improve customer retention.

Output:
```
Logistic Regression:
              precision    recall  f1-score   support

           0       0.80      1.00      0.89      1039
           1       0.00      0.00      0.00       261

    accuracy                           0.80      1300
   macro avg       0.40      0.50      0.44      1300
weighted avg       0.64      0.80      0.71      1300


Random Forest:
              precision    recall  f1-score   support

           0       0.80      1.00      0.89      1039
           1       0.00      0.00      0.00       261

    accuracy                           0.80      1300
   macro avg       0.40      0.50      0.44      1300
weighted avg       0.64      0.80      0.71      1300
```

## 6.5 Interpretation of Confusion Matrices and Classification Reports
The results show that both Logistic Regression and Random Forest perform poorly at identifying customers who are likely to churn, despite achieving approximately 80% accuracy.

### 6.5.1 Confusion matrix interpretation
| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| True Negative (TN) | 1,039 | 1,038 |
| False Positive (FP) | 0 | 1 |
| False Negative (FN) | 261 | 261 |
| True Positive (TP) | 0 | 0 |

Key observations:
- **True Negatives**: Both models correctly identified almost all customers who did not churn.
- **False Positives**: Logistic Regression made no false-positive predictions, while Random Forest made one.
- **False Negatives**: Both models failed to identify all 261 customers who actually churned.
- **True Positives**: Neither model correctly identified a single churned customer.
This means both models are effectively predicting every customer as a non-churner.

### 6.5.2 Classification report interpretation
For class 1 (churned customers):
- Precision = 0.00: Neither model successfully identified any customer as a churner, so there were no correct positive predictions.
- Recall = 0.00: Both models failed to identify any of the 261 actual churned customers.
- F1-score = 0.00: The models have no effectiveness in identifying churned customers based on this metric.

### 6.5.3 Why is accuracy still approximately 80%?
The dataset is imbalanced:
- 79.95% of customers did not churn.
- 20.05% of customers churned.
Because most customers belong to class 0, a model that predicts every customer as a non-churner can still achieve approximately 80% accuracy.

Therefore, accuracy alone gives a misleading impression of model performance.

# Conclusion
Both models achieved approximately 80% accuracy, but neither successfully identified churned customers. Logistic Regression and Random Forest therefore have limited practical value for customer retention because they fail to detect customers who may leave the company.

Recall for class 1 is particularly important in this project because telecom companies need to identify potential churners so that they can take preventive retention actions.

The next step would be to investigate why the models predict almost exclusively class 0, then consider techniques such as class weighting, resampling, or adjusting the classification threshold. These improvements should be evaluated using class 1 recall, precision, and F1-score rather than accuracy alone.
