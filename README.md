# Linear Regression
Description of Linear Regression



What is a Regression

In Regression, we plot a graph between the variables which best fit the given data points. The machine learning model can deliver predictions regarding the data. In naïve words, “Regression shows a line or curve that passes through all the data points on a target-predictor graph in such a way that the vertical distance between the data points and the regression line is minimum.” It is used principally for prediction, forecasting, time series modeling, and determining the causal-effect relationship between variables.



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

Linear Regression 1
The above graph presents the linear relationship between the dependent variable and independent variables. When the value of x (independent variable) increases, the value of y (dependent variable) is likewise increasing. The red line is referred to as the best fit straight line. Based on the given data points, we try to plot a line that models the points the best.
To calculate best-fit line linear regression uses a traditional slope-intercept form.

#### Linear Regression equation

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

* Linear Regression positive
* Negative Linear Relationship

If the dependent variable decreases on the Y-axis and the independent variable increases on the X-axis, such a relationship is called a negative linear relationship.

#### Linear Regression negative
The goal of the linear regression algorithm is to get the best values for a0 and a1 to find the best fit line. The best fit line should have the least error means the error between predicted values and actual values should be minimized.

### Cost function
The cost function helps to figure out the best possible values for a0 and a1, which provides the best fit line for the data points.

Cost function optimizes the regression coefficients or weights and measures how a linear regression model is performing. The cost function is used to find the accuracy of the mapping function that maps the input variable to the output variable. This mapping function is also known as the Hypothesis function.

In Linear Regression, Mean Squared Error (MSE) cost function is used, which is the average of squared error that occurred between the predicted values and actual values.

By simple linear equation y=mx+b we can calculate MSE as:

Let’s y = actual values, yi = predicted values

### Linear Regression MSE
Using the MSE function, we will change the values of a0 and a1 such that the MSE value settles at the minima. Model parameters xi, b (a0,a1) can be manipulated to minimize the cost function. These parameters can be determined using the gradient descent method so that the cost function value is minimum.

### Gradient descent 
Gradient descent is a method of updating a0 and a1 to minimize the cost function (MSE). A regression model uses gradient descent to update the coefficients of the line (a0, a1 => xi, b) by reducing the cost function by a random selection of coefficient values and then iteratively update the values to reach the minimum cost function.

### Linear Regression gradient Descent
Imagine a pit in the shape of U. You are standing at the topmost point in the pit, and your objective is to reach the bottom of the pit. There is a treasure, and you can only take a discrete number of steps to reach the bottom. If you decide to take one footstep at a time, you would eventually get to the bottom of the pit but, this would take a longer time. If you choose to take longer steps each time, you may get to sooner but, there is a chance that you could overshoot the bottom of the pit and not near the bottom. In the gradient descent algorithm, the number of steps you take is the learning rate, and this decides how fast the algorithm converges to the minima.

### Learning Rate
To update a0 and a1, we take gradients from the cost function. To find these gradients, we take partial derivatives for a0 and a1.

Partial derivatives Linear Regression
linear regression cost function 2
linear regression cost function 3
The partial derivates are the gradients, and they are used to update the values of a0 and a1. Alpha is the learning rate.

Impact of different values for learning rate
Impact of different values for learning rate
Source : mygreatleaning.com

The blue line represents the optimal value of the learning rate, and the cost function value is minimized in a few iterations. The green line represents if the learning rate is lower than the optimal value, then the number of iterations required high to minimize the cost function. If the learning rate selected is very high, the cost function could continue to increase with iterations and saturate at a value higher than the minimum value, that represented by a red and black line.

The main function to calculate values of coefficients

1 Initialize the parameters.
2 Predict the value of a dependent variable by given an independent variable.
3 Calculate the error in prediction for all data points.
4 Calculate partial derivative w.r.t a0 and a1.
5 Calculate the cost for each number and add them.
6 Update the values of a0 and a1.
