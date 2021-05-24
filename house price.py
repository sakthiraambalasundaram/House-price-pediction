#Importing the dependincies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

#importing the boston house price data:

house_price_dataset= sklearn.datasets.load_boston()
print(house_price_dataset)

#Loading the dataset to pandas dataframe.
house_price_dataframe=pd.DataFrame(house_price_dataset.data,
                                   columns=house_price_dataset.feature_names)

print(house_price_dataframe)
#print first 5rows of our dataframes

house_price_dataframe.head()

#add the target columns to the dataframe

house_price_dataframe['price']=house_price_dataset.target

house_price_dataframe.head()

#checking the no.of.rows and columns in the dataframe.

house_price_dataframe.shape

#checking for missing values
house_price_dataframe.isnull().sum()

#statistical measure of the dataset.
house_price_dataframe.describe()

#understanding the correlation between various features in the dataset:
 
correlation=house_price_dataframe.corr()

 #constructing heatmap to aunderstand the correlation:
        
plt.figure(figsize=(10,10))
sns.heatmap(correlation, square=True, cbar=True, fmt='.1f', annot=True, 
            annot_kws={'size':8},cmap='Reds')

#splitting the data and target:
x=house_price_dataframe.drop(['price'],axis=1)
y=house_price_dataframe['price']
print(x)
print(y)

#splitting the data into training data and test data:
 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
print(x.shape,x_train.shape,x_test.shape)

#model training
#XGBoost Regressor
#loading the model

model=XGBRegressor()

#training the model with x_train
model.fit(x_train,y_train)

#evaluation
#prediction on training data:

training_data_prediction=model.predict(x_train)
print(training_data_prediction)

# R squared error:
    
score_1=metrics.r2_score(y_train,training_data_prediction)

#mean absolute error:

score_2=metrics.mean_absolute_error(y_train,training_data_prediction)

print("R squared error:",score_1)
print("Mean absolute error:",score_2)

#visualization the actual price and predected prices:
    
plt.scatter(y_train,training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

#predection on test data
#accuracy for predection on test data:

test_data_predection=model.predict(x_test)
    
#mean absolute error:

score_2=metrics.mean_absolute_error(y_train,training_data_prediction)

print("R squared error:",score_1)
print("Mean absolute error:",score_2)