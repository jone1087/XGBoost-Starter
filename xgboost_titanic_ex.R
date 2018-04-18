# Titanic Data Set starter code
# taken and adapted from https://github.com/avinavneer/XGBoost-Starter

# live link for R code and data:
# https://github.com/jone1087/XGBoost-Starter/blob/master/Titanic XGBoost.R

# remove objects
rm(list = ls())
options(stringAsFactors = FALSE)

# Install Packages if you don't have them...
# install.packages("dplyr")
# install.packages("rpart")
# install.packages("randomForest")
# install.packages("e1071")
# install.packages("xgboost")
# install.packages("caret")

# call packages for this session
library(dplyr)
library(rpart)
library(randomForest)
library(e1071)
library(xgboost)
library(caret)

# set directory where the files you will be reading in are stored
# and where files you write will be saved to
setwd("~/Desktop/Projects/Active/SIOP 2018/ML R Tutorial/XGBoost-Starter")

# Combining data sets
train <- read.csv("Data/train.csv")
test  <- read.csv("Data/test.csv")

# test data does not have 'Survived' variable; we will create one so we
# can combine the two data sets into on
test$Survived <- NA
data  <- rbind(train, test)
str(data)

# make PClass a factor
data$Pclass <- as.factor(data$Pclass)

# check for Missing Values
colSums(is.na(data))
table(data[, 'Embarked'])

# Imputing common values in rows missing Fare and Embarked values 
# only a couple values are missing from these two columns so it makes 
# sense to use a basic imputation
data$Fare[is.na(data$Fare)]        <- median(data$Fare, na.rm = TRUE)
data$Embarked[data$Embarked == ''] <- 'S'

# create a dataset for rows that include a value for age
data_1 <- data[!is.na(data$Age), ]

# Creating a decision tree for predicting the missing values of the Age variable
age_model <- rpart(Age ~ SibSp + Parch + Pclass, data_1)
data$Age[is.na(data$Age)] <- predict(age_model, newdata = data[is.na(data$Age), ])


# Pick some predictor variables/features to use 
# Using the dummyVars function of the caret package for One Hot Encoding 
dummy_train <- dummyVars(~ Pclass + Sex + Embarked, data = data)
data_train  <- predict(dummy_train, newdata = data); head(data_train)

# create the predictor set that you will use - needs to be a matrix for xgboost
predictor_set <- cbind(data[, c('Age', 'SibSp', 'Parch', 'Fare')], data_train)
train_pred    <- as.matrix(predictor_set[1:891, ])
train_label   <- train[, "Survived"]; head(train_label)
test_pred     <- predictor_set[892:1309, ]


# Cross validation!
xgb_matrix   <- xgb.DMatrix(train_pred, label = train_label)
xgb_cv_mod_1 <- xgb.cv(data      = xgb_matrix, 
                       max_depth = 20,
                       eta       = 0.01,
                       nrounds   = 100,
                       nfold     = 10,
                       objective = 'binary:logistic',
                       metrics   = 'auc',
                       verbose   = 0)

# do more cross-validation with different parameter combinations to 
# see what happens to the AUC; also do cross-validation when adding/testing
# new features

# After doing a cross-validation to tune the parameters, now we actually
# build the model on the full data set
xgb_mod <- xgboost(data        = xgb_matrix,
                   max_depth   = 20,
                   eta         = 0.01,
                   nrounds     = 100,
                   objective   = 'binary:logistic',
                   eval_metric = 'auc',
                   verbose     = 0)

# make predictions on the test data
xgb_matrix_test <- xgb.DMatrix(as.matrix(test_pred))
predict_probs   <- predict(xgb_mod, newdata = xgb_matrix_test)
predict_01      <- as.numeric(predict_probs > 0.5)
table(predict_01)

# save out a file to upload to kaggle!
write.csv(x         = data.frame(PassengerId = test$PassengerId,
                                 Survived    = predict_01),
          file      = paste0('MyTeam_Submission_1_', 
                              format(Sys.time(), "%m%d%Y", '.csv')),
          row.names = FALSE)


