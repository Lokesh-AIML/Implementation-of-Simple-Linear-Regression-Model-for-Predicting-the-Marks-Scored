# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:LOKESHWARAN S
RegisterNumber:  212224240080
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
df.head()

<img width="261" height="252" alt="image" src="https://github.com/user-attachments/assets/e0b02337-91f4-4bf4-9092-7d5172fc018f" />

df.tail()

<img width="277" height="256" alt="image" src="https://github.com/user-attachments/assets/2e431aaf-63c7-4535-ad57-6ec2612d0e56" />

Array value of X

<img width="259" height="574" alt="image" src="https://github.com/user-attachments/assets/31c2e44b-7201-4f55-9018-9f3ebabfb0ef" />

Array value of Y

<img width="788" height="81" alt="image" src="https://github.com/user-attachments/assets/82708cbc-b074-493c-8e31-eda9ebee1bfe" />

Values of Y prediction

<img width="814" height="73" alt="image" src="https://github.com/user-attachments/assets/e27e7580-d9d4-465b-9c4a-47258d96ac27" />

Array values of Y test

<img width="499" height="58" alt="image" src="https://github.com/user-attachments/assets/98199d67-6a1d-488a-8ba4-c770c50584e3" />

Training set Graph

<img width="851" height="594" alt="image" src="https://github.com/user-attachments/assets/a566a71b-65ea-47f0-ab8f-46c3d1710b79" />

Test set Graph

<img width="800" height="586" alt="image" src="https://github.com/user-attachments/assets/8a8db566-be29-4c40-aaec-b813ccdaba07" />

Values of MSE, MAE and RMSE

<img width="343" height="91" alt="image" src="https://github.com/user-attachments/assets/41e2774d-919d-4bde-bdc7-0058b1656370" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
