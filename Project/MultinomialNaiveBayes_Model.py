import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, Y_train)
Y_pred = mnb.predict(X_test)
acc_mnb = round(mnb.score(X_train, Y_train) * 100, 2)
print("-----------------------------------")
print("  Multinomial Naive Bayes Model  ")
print("-----------------------------------")
print("Multinomial Naive Bayes Accuracy =",round(acc_mnb,2,), "%")
print(Y_pred.shape)
print(Y_pred)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)
