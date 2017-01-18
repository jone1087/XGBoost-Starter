# XGBoost-Starter
## A starter in XGBoost using Kaggle's Titanic dataset 
An introduction to the powerful XGBoost model in R. The Titanic dataset from Kaggle has been used. A description of the basics and working of the model can be found at http://xgboost.readthedocs.io/en/latest/model.html

Running an XGBoost model requires a bit of preprocessing. One of the important things to note is that it takes in only numeric values. Therefore, all the categorical values need to be changed to numeric dummy variables- often by one-hot encoding. Here, the dummyVars function from the caret package has been used to create such dummy class variables. Also, the sparse matrix then created can be converted to a dgCMatrix. Here, they haven't been converted to that format, but the data set has been separated into data and labels. Apart from XGBoost, a Random Forest model has also been used. 

Some further modifications and improvements to the code, including cross validation to determine the hyper-parameters, will follow.
