# Project from scratch- Titanic 

library(dplyr)
library(rpart)
library(randomForest)
library(e1071)
library(xgboost)
library(caret)

rm(list=ls())

setwd("C://Users/Avinav/Desktop/Data Science/Kaggle/Titanic")

# Combining data sets
train = read.csv("train.csv")
test = read.csv("test.csv")
test$Survived = 0
data = rbind(train,test)
str(data)

# train$Survived = as.factor(train$Survived)

data$Survived = as.factor(data$Survived)
data$Pclass = as.factor(data$Pclass)

colSums(is.na(data))

# Imputing common values in rows missing Fare and Embarked values 
data$Fare[is.na(data$Fare)] = median(data$Fare,na.rm = TRUE)
data$Embarked[data$Embarked==''] = 'S'


data_1 = data %>%
  filter(!is.na(Age))

# Creating a decision tree for predicting the missing values of the Age variable

age_model = rpart(Age~SibSp+Parch+Pclass, data_1)
data$Age[is.na(data$Age)] = predict(age_model,newdata = data[is.na(data$Age),])

# Splitting into Training and Test sets
train = data[1:891,]
test = data[892:1309,-2]


# Running a Random Forest model
set.seed(1)
model_rf = randomForest(Survived~Pclass+Sex+Age+SibSp+Parch+
                        Fare+Embarked,data = train,mtry=3,ntree=500)

model_rf
varImpPlot(model_rf)
attributes(model_rf)
model_rf$importance
model_rf$mtry
model_rf$ntree
survival_pred = predict(model_rf,newdata = test)
table(survival_pred)


# Running an XG Boost Model
names(train)
data.xg = train[,c(3,5,6,7,8,10,12)]
label.xg = train[,"Survived"]

names(test)
test.data.xg = test[,c(2,4,5,6,7,9,11)]

# Using the dummyVars function of the caret package for One Hot Encoding 
dummy.train = dummyVars(~Pclass+Sex+Embarked, data = data.xg)
dummy.train
data.train <- predict(dummy.train,data.xg)

train.set = cbind(data.xg,data.train)
names(train.set)
train.set <- train.set[,c(3,4,5,6,8:16)]
train.set <- as.matrix(train.set)

dummy.test = dummyVars(~Pclass+Sex+Embarked, data = test.data.xg)
dummy.test
data.test <- predict(dummy.test,test.data.xg)
test.set = cbind(test.data.xg,data.test)
names(test.set)
test.set <- test.set[,c(3,4,5,6,8:16)]
test.set <- as.matrix(test.set)

# XGBoost model with the specified hyper parameters 

xgb.model = xgboost(data = train.set, label = as.matrix(label.xg),max_depth=20, 
                    nrounds = 500, eta = 0.01, verbose = 1, 
                    objective = "binary:logistic",eval_metric ="auc" )

xgb.model
survival_pred = predict(xgb.model,newdata = test.set)
prediction = as.numeric(survival_pred>0.5)
table(prediction)


