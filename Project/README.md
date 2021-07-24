# Description: How we Achieved each Task?


In this project, we implement four classifications techniques


 1.  Linear Classification 
 2.  SVM 
 3.  KNN 

In each technique, we are dropping some coluns of name wich are not useable, so we used 3 different columns in each model.
   1. Firstly we drop some columns from train.csv and test.csv same as we did in Assignment_03.
   2. Then we apply crossvalidation or KFold technique by using 3 columns from train.csv and test.csv.
   3. Finally we implement models and get accuracy and predicted values.




# CV Score of Each Three Techniques: #

## 1. KNN: ##

1. In this model, it will find nearest neighbor on K-Value which is in the odd after get the sqrt on yTest (from CV).
2. After the crossvalidation on train.csv and test.csv it separfates the test data upto 20% or 30% and train data upto 80% or 70%(we have changed it randomly).
3. After the application of KNN model we achieved a score of 0.842.



## 2. SVM: ##

1. This model is different from other because it does not learn on the characteristics not like other models learn.
2. After the crossvalidation on train.csv and test.csv it separfates the test data upto 20% or 30% and train data upto 80% or 70%(we have changed it randomly).
3. After the application of SVM model we achieved a score of 0.78.


## 3. Linear Classification: ##

1. This model used to minimize the sum of square between  the observed and target in the data set and the target predicted by the linear approximation.
2. We are using Logistic Regression.
3. After the crossvalidation on train.csv and test.csv it separfates the test data upto 20% or 30% and train data upto 80% or 70%(we have changed it randomly).
4. After the application of LC model we achieved a score of 0.80.





# Description: Important part of .py file: #


### Convolution Part: ###

1. In this part, we are applying 5x5,7x7,9x9 convolution to map on our 42000 data, It will help to predict and get the filtered image/label.
2. Explaining about its working, Firstly, we can break our 784 columns into 28x28 and create 2D Array and iterate on array filter will push into it.


### Models Part: ###

1. We implement Three techniques and on these techniques, we are applying crossvalidation to separates training or testing data, to get the best/good score.
2. But according to our views to work on this phase, we achieve best score on KNN.









## KNN MODEL SCREENSHOT ON KAGGLE:
![KNN_SS](https://user-images.githubusercontent.com/61589430/126374521-258e8ef6-bda6-479d-b87d-71f9a19073c7.JPG)

### KNN MODEL ACCURACY SCREENSHOT ON CODE:
![KNN_accuracy](https://user-images.githubusercontent.com/61589430/126375551-bb086e5b-204b-40dd-9344-e1531089b9e2.JPG)


## K-FOLD CROSS VAILDATION MODEL SCREENSHOT ON KAGGLE:
![KFoldCross_SS](https://user-images.githubusercontent.com/61589430/126375613-87e6a651-4b25-4367-aeda-b857daea7f98.JPG)

### K-FOLD CROSS VAILDATION MODEL ACCURACY SCREENSHOT ON CODE:
![K-Fold_accuracy](https://user-images.githubusercontent.com/61589430/126375657-fc4ac019-6fc1-44be-a8db-10a6891267e9.JPG)


## LC MODEL SCREENSHOT ON KAGGLE:
![LC_SS](https://user-images.githubusercontent.com/49693169/126375868-ee00b2e3-d3d6-4dc7-8f85-5a9cd85f8381.JPG)

### LC MODEL ACCURACY SCREENSHOT ON CODE:
![LC_accuracy](https://user-images.githubusercontent.com/49693169/126375902-c236e7e4-9406-4d94-af54-27d8d187922a.JPG)


## SVM MODEL SCREENSHOT ON KAGGLE:
![SVM_SS](https://user-images.githubusercontent.com/49693169/126375933-e5c9da6b-691c-4e57-a8f5-bb83967ddd4c.JPG)

### SVM MODEL ACCURACY SCREENSHOT ON CODE:
![SVM_accuracy](https://user-images.githubusercontent.com/49693169/126375957-f20a478d-8fab-41f5-9afd-712511b7463e.JPG)


## MULTINOMIAL NAIVE BAYES MODEL SCREENSHOT ON KAGGLE:
![Multinomial_SS](https://user-images.githubusercontent.com/49693169/126376014-298a66bb-888d-4ff1-8c18-9baa39b15979.JPG)

### MULTINOMIAL NAIVE BAYES MODEL ACCURACY SCREENSHOT ON CODE:
![Multinomial_accuracy](https://user-images.githubusercontent.com/49693169/126376027-3898dec6-a2d9-4086-b93a-0c7eeadcba74.JPG)

# COMPLETE SUBMISSION SCREENSHOT ON KAGGLE:
![CompleteSubmissions](https://user-images.githubusercontent.com/61589430/126376216-e71e654b-178e-4d38-b2a6-680676d8250a.JPG)
