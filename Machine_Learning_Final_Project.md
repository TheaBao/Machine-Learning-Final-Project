# Practical Machine Learning Final Project
Date: 05/23/2015

##Background Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement ??? a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

Reference : Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human ’13) . Stuttgart, Germany: ACM SIGCHI, 2013.

Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions:

exactly according to the specification (Class A),
throwing the elbows to the front (Class B),
lifting the dumbbell only halfway (Class C),
lowering the dumbbell only halfway (Class D)
throwing the hips to the front (Class E).

##Data Sources
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project comes from this original source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

##Objective
The goal of your project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details. 

##Reproduceablity
This project requires installing and loading the following packages in your working environment:

```r
#install.packages("caret") first if not yet installed
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
#install.packages("rpart") first if not yet installed
library(rpart)
#install.packages("randomForest") first if not yet installed
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
#install.packages("e1071") first if not yet installed
library(e1071)
```
Also, to obtain the same result you should set a pseudo random seed as:

```r
set.seed(413)
```

##Getting and Cleaning the Data

```r
##Set the Url's for training dataset and testing dataset accordingly
train_Url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_Url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

##Detect whether data files are in working directory already
##Load the data from local or online sources to memory
##At the same time, set all missing values "#DIV/0!" or "" or "NA" to "NA" only
if (file.exists("pml-training.csv")) {
        train <- read.csv("pml-training.csv", na.strings=c("#DIV/0!","","NA"))
} else { 
        download.file(train_Url,"pml-training.csv")
        train <- read.csv(url(train_Url), na.strings=c("#DIV/0!","","NA"))
        }                           

if (file.exists("pml-testing.csv")) {
        test <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!","","NA"))
} else { 
        download.file(test_Url,"pml-testing.csv")
        test <- read.csv(url(test_Url), na.strings=c("#DIV/0!","","NA"))
        } 
##Check each dimensions
dim(train)
```

```
## [1] 19622   160
```

```r
dim(test)
```

```
## [1]  20 160
```

```r
##Column names' coherence Check
all.equal(colnames(test)[1:length(colnames(test))-1], colnames(train)[1:length(colnames(train))-1])
```

```
## [1] TRUE
```

```r
##Delete columns with all "NA"'s
train<-train[,colSums(is.na(train)) == 0]
test<-test[,colSums(is.na(test)) == 0]

##Delete the first columns (ID) of datasets to better perform ML Algorithms:
train<-train[c(-1)]
test <- test[c(-1)]

##Check each dimensions again
dim(train)
```

```
## [1] 19622    59
```

```r
dim(test)
```

```
## [1] 20 59
```

##Partioning the training dataset
The training dataset should be partionned into two subsets to perform cross-validation:

```r
##Partioning into two subsets: 70% for sub_train and 30% for sub_test
subsets <- createDataPartition(y=train$classe, p=0.7, list=FALSE)
sub_train <- train[subsets, ] 
sub_test <- train[-subsets, ]
##Check each dimensions again
dim(sub_train)
```

```
## [1] 13737    59
```

```r
dim(sub_test)
```

```
## [1] 5885   59
```

##Prediction Model with Machine Learning Algorithm - Random Forest
Random Forest method is adopted in this project for its higher accuracy when dealing a great number of predictor variables. 

```r
model <- randomForest(classe ~. , data=sub_train, method="class")

##Cross Validation
#Using the model to predict activity quality from activity monitors:
prediction <- predict(model, sub_test, type = "class")

#Testing the results with partitioned sub_test dataset:
confusionMatrix(prediction, sub_test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    0    0    0
##          B    1 1138    1    0    0
##          C    0    1 1024    2    0
##          D    0    0    1  962    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.999           
##                  95% CI : (0.9978, 0.9996)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9987          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9991   0.9981   0.9979   1.0000
## Specificity            1.0000   0.9996   0.9994   0.9998   1.0000
## Pos Pred Value         1.0000   0.9982   0.9971   0.9990   1.0000
## Neg Pred Value         0.9998   0.9998   0.9996   0.9996   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1934   0.1740   0.1635   0.1839
## Detection Prevalence   0.2843   0.1937   0.1745   0.1636   0.1839
## Balanced Accuracy      0.9997   0.9994   0.9987   0.9989   1.0000
```

##Check for Accuracy and Out of Sample Error

```r
##Calculate the Accuracy 
postResample(prediction, sub_test$classe)
```

```
##  Accuracy     Kappa 
## 0.9989805 0.9987104
```

```r
##Calculate the Out of Sample Error
1 - as.numeric(confusionMatrix(sub_test$classe, prediction)$overall[1])
```

```
## [1] 0.001019541
```
From which we can see the accuracy for this prediction is 99.9%,
and its out of sample error is 0.1%.

##Predicting for Test Dataset and Subimmting Online

```r
##Clean the testing data again to keep the same type as training data
clean<-colnames(sub_train[, -59])
test<-test[clean]
for (i in 1:length(test) ) {
        for(j in 1:length(sub_train)) {
        if( length(grep(names(sub_train[i]), names(test)[j])) ==1)  {
            class(test[j])<-class(sub_train[i])
        }      
    }      
}
##Make sure Coertion can work on testing data
testClean <- rbind(sub_train[2, -59] , test)
##Remove row2 since it does not carry meanings any more
testClean <- testClean[-1,]

##Apply this model to the original testing dataset 
prediction_test <- predict(model, testClean, type="class")

##Generate the “problem_id_x.txt” file in working directory.
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(prediction_test)
```
