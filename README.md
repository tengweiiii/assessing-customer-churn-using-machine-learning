# assessing-customer-churn-using-machine-learning
Dive into India's telecom sector to analyze customer churn. Utilize pandas and machine learning to study datasets from top telecom firms, revealing demographic and usage patterns. Predict customer retention, merging data analysis and predictive modeling to sharpen your data science expertise.

The telecommunications (telecom) sector in India is rapidly changing, with more and more telecom businesses being created and many customers deciding to switch between providers. "Churn" refers to the process where customers or subscribers stop using a company's services or products. Understanding the factors that influence keeping a customer as a client in predicting churn is crucial for telecom companies to enhance their service quality and customer satisfaction. As the data scientist on this project, you aim to explore the intricate dynamics of customer behavior and demographics in the Indian telecom sector in predicting customer churn, utilizing two comprehensive datasets from four major telecom partners: Airtel, Reliance Jio, Vodafone, and BSNL:

- `telecom_demographics.csv` contains information related to Indian customer demographics:

| Variable             | Description                                      |
|----------------------|--------------------------------------------------|
| `customer_id `         | Unique identifier for each customer.             |
| `telecom_partner `     | The telecom partner associated with the customer.|
| `gender `              | The gender of the customer.                      |
| `age `                 | The age of the customer.                         |
| `state`                | The Indian state in which the customer is located.|
| `city`                 | The city in which the customer is located.       |
| `pincode`              | The pincode of the customer's location.          |
| `registration_event` | When the customer registered with the telecom partner.|
| `num_dependents`      | The number of dependents (e.g., children) the customer has.|
| `estimated_salary`     | The customer's estimated salary.                 |

- `telecom_usage` contains information about the usage patterns of Indian customers:

| Variable   | Description                                                  |
|------------|--------------------------------------------------------------|
| `customer_id` | Unique identifier for each customer.                         |
| `calls_made` | The number of calls made by the customer.                    |
| `sms_sent`   | The number of SMS messages sent by the customer.             |
| `data_used`  | The amount of data used by the customer.                     |
| `churn`    | Binary variable indicating whether the customer has churned or not (1 = churned, 0 = not churned).|

Does Logistic Regression or Random Forest produce a higher accuracy score in predicting telecom churn in India?

* Load the two CSV files into separate DataFrames. Merge them into a DataFrame named `churn_df`. Calculate the proportion of customers who have churned, and identify the categorical variables in `churn_df`.
* Convert categorical features in `churn_df` into `features_scaled`. Perform feature scaling separating the appropriate features and scale them. Define your scaled features and `target` variable for the churn prediction model.
* Split the processed data into training and testing sets giving names of `X_train`, `X_test`, `y_train`, and `y_test` using an 80-20 split, setting a random state of `42` for reproducibility.
* Train Logistic Regression and Random Forest Classifier models, setting a random seed of `42`. Store model predictions in `logreg_pred` and `rf_pred`.
Assess the models on test data. Assign the model's name with higher accuracy (`"LogisticRegression"` or `"RandomForest"`) to `higher_accuracy`.

```
# Import libraries and methods/functions
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Start your code here!
```
