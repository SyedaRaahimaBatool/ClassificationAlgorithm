import numpy as np
import pandas as pdfrom sklearn.svm import LinearSVC
# SVM Model
linear_svm = LinearSVC()
linear_svm.fit(X_train, Y_train)
Y_pred = linear_svm.predict(X_test)
acc_linear_svc = round(linear_svm.score(X_train, Y_train) * 100, 2)
print("--------------")
print("  SVM Model  ")
print("--------------")
print("SVM Accuracy =", round(acc_linear_svc,2,), "%")
print(Y_pred.shape)
print(Y_pred)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('Submission_Of_SVM.csv', index=False)

linear_svm = LinearSVC()
scores = cross_val_score(linear_svm, X_train, Y_train, cv=10, scoring = "accuracy")
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
print("Scores:\n", pd.Series(scores))
