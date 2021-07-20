import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
# K-Fold Cross Validation
Mul = MultinomialNB()
print("-----------------------------")
print("  K-Fold Cross Validation  ")
print("-----------------------------")
scores = cross_val_score(Mul, X_train, Y_train, cv=10, scoring = "accuracy")
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
print("Scores:\n", pd.Series(scores))
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('Submission_Of_KFoldCross.csv', index=False)
