import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the credit card data
data = pd.read_csv("creditcard.csv")

# Normalize the "Amount" column
scaler = StandardScaler()
data['normAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Class', axis=1), data['Class'], test_size=0.25, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_mat)
print("Accuracy Score:\n", accuracy_score(y_test, y_pred))

# Calculate the number of non-fraudulent and fraudulent transactions
num_non_fraud = conf_mat[0][0] + conf_mat[0][1]
num_fraud = conf_mat[1][0] + conf_mat[1][1]
print("Number of non-fraudulent transactions:", num_non_fraud)
print("Number of fraudulent transactions:", num_fraud)
