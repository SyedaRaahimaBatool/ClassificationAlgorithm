import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# KNN Model
KNN = KNeighborsClassifier(n_neighbors = 3)
KNN.fit(X_train, Y_train)
Y_pred = KNN.predict(X_test)
acc_KNN = round(KNN.score(X_train, Y_train) * 100, 2)
print("--------------")
print("  KNN Model  ")
print("--------------")
print("KNN Accuracy =",round(acc_KNN,2,), "%")
print(Y_pred.shape)
print(Y_pred)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('Submission_Of_KNN.csv', index=False)

KNN = KNeighborsClassifier(n_neighbors = 3)
scores = cross_val_score(KNN, X_train, Y_train, cv=10, scoring = "accuracy")
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
print("Scores:\n", pd.Series(scores))

