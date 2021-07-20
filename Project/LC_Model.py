import numpy as np
import pandas as pd

# Algorithms
from sklearn.linear_model import LogisticRegression

# Linear Classifiers
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("--------------")
print("  LC Model  ")
print("--------------")
print("Linear Classifiers Accuracy =",round(acc_log,2,), "%")
print(Y_pred.shape)
print(Y_pred)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('Submission_Of_LinearClassifiers.csv', index=False)

logreg = LogisticRegression()
scores = cross_val_score(logreg, X_train, Y_train, cv=10, scoring = "accuracy")
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
print("Scores:\n", pd.Series(scores))

