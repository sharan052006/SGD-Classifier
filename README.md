# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries like sklearn, pandas, and matplotlib.

2.Load the Iris dataset and convert it into a DataFrame.

3.Split the data into features (X) and target (Y).

4.Divide data into training and testing sets.

5.Train the model using SGDClassifier on the training data.

6.Predict and evaluate using accuracy score and confusion matrix.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Sharan.I
RegisterNumber:  212224040308
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
iris = load_iris()

# Create pandas dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())
# Split the data into features (X) and target (Y)
x = df.drop('target', axis=1)
y = df['target']
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Create SGD Classifier with default parameters
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

# Train the classifier on the training data
sgd_clf.fit(x_train, y_train)
# Make predictions on the testing data
y_pred = sgd_clf.predict(x_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
# Calculate the confusion matrix
cf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cf)
*/
```

## Output:

![image](https://github.com/user-attachments/assets/5452aa4e-3029-43d4-a7a3-b67afaffce20)

![image](https://github.com/user-attachments/assets/e6d22720-f196-4fb5-8c3c-b969b25f4725)

![image](https://github.com/user-attachments/assets/a8055e92-d650-407d-9765-c946e8ead2f3)

![image](https://github.com/user-attachments/assets/03ac9826-3a05-4ae4-9a52-8505d2176f05)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
