# Linear Regression
Description of Linear Regression




## Introduction

Regression is a supervised learning technique that supports finding the correlation among variables. A regression problem is when the output variable is a real or continuous value.

 In this article, we will understand the following concepts:

##### What is a Regression?

###### Types of a Regression.

##### What is the mean of Linear regression and the importance of Linear regression?

##### Importance of cost function and gradient descent in a Linear regression.

##### Impact of different values for learning rate.

 ##### Implement use case of Linear regression with python code.
 

## What is a Regression

In Regression, we plot a graph between the variables which best fit the given data points. The machine learning model can deliver predictions regarding the data. In naïve words, “Regression shows a line or curve that passes through all the data points on a target-predictor graph in such a way that the vertical distance between the data points and the regression line is minimum.” It is used principally for prediction, forecasting, time series modeling, and determining the causal-effect relationship between variables.

 

## Types of Regression models

 * Linear Regression
 * Polynomial Regression
 * Logistics Regression
 

### Linear Regression
Linear regression is a quiet and simple statistical regression method used for predictive analysis and shows the relationship between the continuous variables. Linear regression shows the linear relationship between the independent variable (X-axis) and the dependent variable (Y-axis), consequently called linear regression. If there is a single input variable (x), such linear regression is called simple linear regression. And if there is more than one input variable, such linear regression is called multiple linear regression. The linear regression model gives a sloped straight line describing the relationship within the variables.

![image](https://editor.analyticsvidhya.com/uploads/72060linear.png)


Linear Regression 1
The above graph presents the linear relationship between the dependent variable and independent variables. When the value of x (independent variable) increases, the value of y (dependent variable) is likewise increasing. The red line is referred to as the best fit straight line. Based on the given data points, we try to plot a line that models the points the best.
To calculate best-fit line linear regression uses a traditional slope-intercept form.

#### Linear Regression equation

![image](https://editor.analyticsvidhya.com/uploads/32826linear1.png)
 
y= Dependent Variable.

x= Independent Variable.

a0= intercept of the line.

a1 = Linear regression coefficient.

#### Need of a Linear regression

As mentioned above, Linear regression estimates the relationship between a dependent variable and an independent variable. Let’s understand this with an easy example:

Let’s say we want to estimate the salary of an employee based on year of experience. You have the recent company data, which indicates that the relationship between experience and salary. Here year of experience is an independent variable, and the salary of an employee is a dependent variable, as the salary of an employee is dependent on the experience of an employee. Using this insight, we can predict the future salary of the employee based on current & past information.

A regression line can be a Positive Linear Relationship or a Negative Linear Relationship.

 

#### Positive Linear Relationship

If the dependent variable expands on the Y-axis and the independent variable progress on X-axis, then such a relationship is termed a Positive linear relationship.

![image](https://editor.analyticsvidhya.com/uploads/11467linear2.png)
* Linear Regression positive
* Negative Linear Relationship

If the dependent variable decreases on the Y-axis and the independent variable increases on the X-axis, such a relationship is called a negative linear relationship.

#### Linear Regression negative

![image](https://editor.analyticsvidhya.com/uploads/35247linear3.png)

The goal of the linear regression algorithm is to get the best values for a0 and a1 to find the best fit line. The best fit line should have the least error means the error between predicted values and actual values should be minimized.

### Cost function


The cost function helps to figure out the best possible values for a0 and a1, which provides the best fit line for the data points.

Cost function optimizes the regression coefficients or weights and measures how a linear regression model is performing. The cost function is used to find the accuracy of the mapping function that maps the input variable to the output variable. This mapping function is also known as the Hypothesis function.

In Linear Regression, Mean Squared Error (MSE) cost function is used, which is the average of squared error that occurred between the predicted values and actual values.

By simple linear equation y=mx+b we can calculate MSE as:

![image](https://editor.analyticsvidhya.com/uploads/59553mse.png)

Let’s y = actual values, yi = predicted values

### Linear Regression MSE
Using the MSE function, we will change the values of a0 and a1 such that the MSE value settles at the minima. Model parameters xi, b (a0,a1) can be manipulated to minimize the cost function. These parameters can be determined using the gradient descent method so that the cost function value is minimum.

## Gradient descent 



Gradient descent is a method of updating a0 and a1 to minimize the cost function (MSE). A regression model uses gradient descent to update the coefficients of the line (a0, a1 => xi, b) by reducing the cost function by a random selection of coefficient values and then iteratively update the values to reach the minimum cost function.

![image](https://editor.analyticsvidhya.com/uploads/68835linear4.png)

### Linear Regression gradient Descent
Imagine a pit in the shape of U. You are standing at the topmost point in the pit, and your objective is to reach the bottom of the pit. There is a treasure, and you can only take a discrete number of steps to reach the bottom. If you decide to take one footstep at a time, you would eventually get to the bottom of the pit but, this would take a longer time. If you choose to take longer steps each time, you may get to sooner but, there is a chance that you could overshoot the bottom of the pit and not near the bottom. In the gradient descent algorithm, the number of steps you take is the learning rate, and this decides how fast the algorithm converges to the minima.

![image](https://editor.analyticsvidhya.com/uploads/97695learn.png)

### Learning Rate
To update a0 and a1, we take gradients from the cost function. To find these gradients, we take partial derivatives for a0 and a1.

![image](https://editor.analyticsvidhya.com/uploads/43974final_dev1.png)
![image](https://editor.analyticsvidhya.com/uploads/47189final_dev2.png)
![image](https://editor.analyticsvidhya.com/uploads/18613final_dev3.png)

Partial derivatives Linear Regression
linear regression cost function 2
linear regression cost function 3
The partial derivates are the gradients, and they are used to update the values of a0 and a1. Alpha is the learning rate.

Impact of different values for learning rate

![image](https://editor.analyticsvidhya.com/uploads/71216Learn_rate.png)
Source : mygreatleaning.com


The blue line represents the optimal value of the learning rate, and the cost function value is minimized in a few iterations. The green line represents if the learning rate is lower than the optimal value, then the number of iterations required high to minimize the cost function. If the learning rate selected is very high, the cost function could continue to increase with iterations and saturate at a value higher than the minimum value, that represented by a red and black line.

The main function to calculate values of coefficients

1 Initialize the parameters.

2 Predict the value of a dependent variable by given an independent variable.

3 Calculate the error in prediction for all data points.

4 Calculate partial derivative w.r.t a0 and a1.

5 Calculate the cost for each number and add them.

6 Update the values of a0 and a1.


## Implementation of Linear Regression Using Python code




In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

#### Importing Libraries
To import necessary libraries for this task, execute the following import statements:


###### import pandas as pd
###### import numpy as np
###### import matplotlib.pyplot as plt
###### %matplotlib inline

##### Dataset

The dataset being used for this example has been made publicly available and can be downloaded from this link:

https://drive.google.com/open?id=1oakZCv7g3mlmCSdv9J8kdSaqO5_6dIOw

Note: This example was executed on a Windows based machine and the dataset was stored in "D:\datasets" folder. You can download the file in a different location as long as you change the dataset path accordingly.

The following command imports the CSV dataset using pandas:

###### dataset = pd.read_csv('D:\Datasets\student_scores.csv')


Now let's explore our dataset a bit. To do so, execute the following script:

###### dataset.shape


After doing this, you should see the following printed out:

###### (25, 2)

This means that our dataset has 25 rows and 2 columns. Let's take a look at what our dataset actually looks like. To do this, use the head() method:

##### dataset.head()


The above method retrieves the first 5 records from our dataset, which will look like this:

## FIGURE


##### dataset.describe()

## FIGURE

And finally, let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

##### dataset.plot(x='Hours', y='Scores', style='o')
##### plt.title('Hours vs Percentage')
##### plt.xlabel('Hours Studied')
##### plt.ylabel('Percentage Score')
##### plt.show()

In the script above, we use plot() function of the pandas dataframe and pass it the column names for x coordinate and y coordinate, which are "Hours" and "Scores" respectively.

The resulting plot will look like this:

### FIGURE

From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.



#### Preparing the Data



Now we have an idea about statistical details of our data. The next step is to divide the data into "attributes" and "labels". Attributes are the independent variables while labels are dependent variables whose values are to be predicted. In our dataset we only have two columns. We want to predict the percentage score depending upon the hours studied. Therefore our attribute set will consist of the "Hours" column, and the label will be the "Score" column. To extract the attributes and labels, execute the following script:

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

The attributes are stored in the X variable. We specified "-1" as the range for columns since we wanted our attribute set to contain all the columns except the last one, which is "Scores". Similarly the y variable contains the labels. We specified 1 for the label column since the index for "Scores" column is 1. Remember, the column indexes start with 0, with 1 being the second column. In the next section, we will see a better way to specify columns for attributes and labels.

Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


The above script splits 80% of the data to training set while 20% of the data to test set. The test_size variable is where we actually specify the proportion of test set.

### Training the Algorithm



We have split our data into training and testing sets, and now is finally the time to train our algorithm. Execute following command:

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

With Scikit-Learn it is extremely straight forward to implement linear regression models, as all you really need to do is import the LinearRegression class, instantiate it, and call the fit() method along with our training data. This is about as simple as it gets when using a machine learning library to train on your data.

In the theory section we said that linear regression model basically finds the best value for the intercept and slope, which results in a line that best fits the data. To see the value of the intercept and slop calculated by the linear regression algorithm for our dataset, execute the following code.

To retrieve the intercept:

print(regressor.intercept_)
The resulting value you see should be approximately 2.01816004143.

For retrieving the slope (coefficient of x):

print(regressor.coef_)

The result should be approximately 9.91065648.

This means that for every one unit of change in hours studied, the change in the score is about 9.91%. Or in simpler words, if a student studies one hour more than they previously studied for an exam, they can expect to achieve an increase of 9.91% in the score achieved by the student previously.

#### Making Predictions
Now that we have trained our algorithm, it's time to make some predictions. To do so, we will use our test data and see how accurately our algorithm predicts the percentage score. To make pre-dictions on the test data, execute the following script:

y_pred = regressor.predict(X_test)


The y_pred is a numpy array that contains all the predicted values for the input values in the X_test series.

To compare the actual output values for X_test with the predicted values, execute the following script:

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df

The output looks like this:

#### FIGURE

Though our model is not very precise, the predicted percentages are close to the actual ones.

Note:

The values in the columns above may be different in your case because the train_test_split function randomly splits data into train and test sets, and your splits are likely different from the one shown in this article.

#### Evaluating the Algorithm

The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For regression algorithms, three evaluation metrics are commonly used:

Mean Absolute Error (MAE) is the mean of the absolute value of the errors. It is calculated as:

Mean Absolute Error

Mean Squared Error (MSE) is the mean of the squared errors and is calculated as:

Mean Squared Error

Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:

Root Mean Squared Error

Luckily, we don't have to perform these calculations manually. The Scikit-Learn library comes with pre-built functions that can be used to find out these values for us.

Let's find the values for these metrics using our test data. Execute the following code:

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

The output will look similar to this (but probably slightly different):

Mean Absolute Error: 4.183859899
Mean Squared Error: 21.5987693072
Root Mean Squared Error: 4.6474476121

You can see that the value of root mean squared error is 4.64, which is less than 10% of the mean value of the percentages of all the students i.e. 51.48. This means that our algorithm did a decent job.


