# -------------------------------------------
# Title:  Linear regression and ridge regression
# Author: Erik Hjalmarsson
# File:   Assignment_2.R
# Date:   2023-12-06
# Last    modified: 2023-12-13
# -------------------------------------------

# Load necessary libraries
library(dplyr)
library(tidyr)
library(caret) # For scaling the data

data = read.csv("parkinsons.csv")

#--------------------------------------------
# Task 1
#--------------------------------------------

set.seed(12345)
n=dim(data)[1]

# Partitioning the data into training and test datasets
id = sample(1:n, floor(n*0.60))
train = data[id,]
test = data[-id,]

# preProcess processes the data and applies an appropriate scaler
scaler = preProcess(train)

# Scaling both datasets with same scaler to keep consistency
trainS = predict(scaler, train)
testS = predict(scaler, test)

#--------------------------------------------
# Task 2
#--------------------------------------------

# Dependent variable motor_UPDRS and independent variable as in the assignment description
# The negative sign before the columns name indicates that we remove that column from the linear regression model
lr_model <- lm(trainS$motor_UPDRS~. -subject. -age -sex -test_time -total_UPDRS, data=trainS)

summary(lr_model)

# Calculating MSE for lr_model
MSE_lr_model <- mean(lr_model$residuals^2)
print(paste("Error for lr_model:",MSE_lr_model))

# Predicting values for test dataset using lr_model
test_hat <- predict(lr_model, newdata=testS)

# Calculating Mean Squared Error of testset
MSE_test <- mean((testS$motor_UPDRS-test_hat)^2)
print(paste("Error for MSE test data:",MSE_test))

# Which coefficients have most weight?
print(lr_model$coefficients)

#--------------------------------------------
# Task 3
#--------------------------------------------

#--------------------#
# A - log-likelihood #
#--------------------#

loglikelihood <- function (theta,sigma){
  y <- as.matrix(trainS$motor_UPDRS)
  x <- as.matrix(trainS %>% select(Jitter...:PPE))
  
  loglikelihood_var = -(nrow(trainS) * 0.5) * log(2 * pi * sigma^2) - (0.5 * sigma^-2)*(sum((x %*% theta - y)^2))
  return(loglikelihood_var)
}

#-----------#
# B - Ridge #
#-----------#

ridge <- function(theta, lambda) {
  sigma <- theta[17]
  theta <- theta[-17]
  return(lambda * sum(theta^2)-loglikelihood(theta, sigma))
}

#--------------#
# C - RidgeOpt #
#--------------#

ridgeopt <- function(lambda){
  return(optim(rep(1,17), fn=ridge,lambda=lambda, method="BFGS"))
}

#-----------------#
# D - DF function #
#-----------------#

DF <- function(lambda){
  x <- as.matrix(trainS %>% select(Jitter...:PPE))
  p <- ncol(x)
  df <- sum(diag(x %*% solve(t(x) %*% x + lambda * diag(p)) %*% t(x)))
  return(df)
}

#--------------------------------------------
# Task 4
#--------------------------------------------

trainS_matrix <- as.matrix(trainS %>% select(Jitter...:PPE))
testS_matrix <- as.matrix(testS %>% select(Jitter...:PPE))

for (i in c(1, 100, 1000)){
  lambda_ridgeopt <- ridgeopt(i)
  predicted_UPDRS <- trainS_matrix %*% as.matrix(lambda_ridgeopt$par[-17])
  MSE_lambda <- mean((trainS$motor_UPDRS-predicted_UPDRS)^2)
  print(paste("MSE_Train with lambda :", i, " = ", MSE_lambda))  
  
  predicted_UPDRS <- testS_matrix %*% as.matrix(lambda_ridgeopt$par[-17])
  MSE_lambda <- mean((testS$motor_UPDRS-predicted_UPDRS)^2)
  print(paste("MSE_Test with lambda :", i, " = ", MSE_lambda))  
  
  print(paste("Df ", i, ": ", DF(i)))
}

# Plot for test Scaled
plot(testS$motor_UPDRS, test_hat, 
     main = "Observed vs Predicted Values",
     xlab = "Observed Values",
     ylab = "Predicted Values")

# Add regression line to the plot
abline(lm(test_hat ~ testS$motor_UPDRS), col = "red")

lambda1_ridgeopt <- ridgeopt(1)
predicted_UPDRS <- testS_matrix %*% as.matrix(lambda_ridgeopt$par[-17])

# Plot for test Scaled
plot(testS$motor_UPDRS, predicted_UPDRS, 
     main = "Observed vs Predicted Values",
     xlab = "Observed Values",
     ylab = "Predicted Values")

# Add regression line to the plot
abline(lm(predicted_UPDRS ~ testS$motor_UPDRS), col = "red")















