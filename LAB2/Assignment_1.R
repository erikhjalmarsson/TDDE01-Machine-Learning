# -------------------------------------------
# Title:  Explicit regularization
# Author: Erik Hjalmarsson
# File:   Assignment_1.R
# Date:   2023-12-06
# Last modified: 2023-12-14
# -------------------------------------------

# Load necessary libraries
library(glmnet)
library(caret)
library(dplyr)
library(tidyr)

#Reading dataframe from csv file
data_import <- read.csv("tecator.csv")

data <- data %>% select(Channel1:Moisture)

#Splitting data into train, validation and test
n <- dim(data)[1]
set.seed(12345)
id <- sample(1:n, floor(n * 0.5))
train <- data[id,]
test <- data[-id,]

#--------------------------------------------
# Task 1
#--------------------------------------------

#Fit the linear regression to the training data
lr_model <- lm(train$Fat~. -Protein - Moisture, data=train)

summary(lr_model)

#Estimates the training and test error as MSE
train_predictions <- predict(lr_model, newdata = train)
train_error <- mean((train$Fat - train_predictions)^2)

test_predictions <- predict(lr_model, newdata = test)
test_error <- mean((test$Fat - test_predictions)^2)

cat("Training Error (MSE):", train_error, "\n")
cat("Test Error (MSE):", test_error, "\n")

#--------------------------------------------
# Task 2
#--------------------------------------------

# Fit the LASSO regression model
lasso_model <- glmnet(train %>% select(Channel1:Channel100),
                      as.matrix(train %>% select(Fat)),
                      family="gaussian",
                      alpha = 1)

# Print the results
print(lasso_model)

# Get the coefficients
coefficients <- coef(lasso_model)

# Extract the values of lambda
lambda_values <- lasso_model$lambda

# Print the coefficients and lambda values
print(coefficients)
print(lambda_values)


#--------------------------------------------
# Task 3
#--------------------------------------------

# Plot the regularization path
plot(lasso_model, xvar = "lambda", label = TRUE, main = "LASSO")

#--------------------------------------------
# Task 4
#--------------------------------------------

# Fit the ridge regression model
ridge_model <- glmnet(train %>% select(Channel1:Channel100),
                      as.matrix(train %>% select(Fat)),
                      alpha = 0,
                      family="gaussian")

# Print the results
print(ridge_model)

plot(ridge_model, xvar = "lambda", label=TRUE, main = "Ridge")

#--------------------------------------------
# Task 5
#--------------------------------------------

# Fit the final LASSO model with the optimal lambda
opt_lasso_model <- cv.glmnet(as.matrix(train %>% select(Channel1:Channel100)),
                             as.matrix(train %>% select(Fat)),
                             alpha = 1,
                             family="gaussian")

# Display the optimal lambda value
optimal_lambda <- opt_lasso_model$lambda.min
cat("Optimal lambda:", optimal_lambda, "\n")

# Plot the dependence of the CV score on log lambda
plot(opt_lasso_model)

# Extract the coefficients for the optimal lambda
lasso_coefficients <- coef(opt_lasso_model, s = optimal_lambda)
# Printing the coefficients that are != 0
print("Lasso coefficients that are non zero:")
print(lasso_coefficients[lasso_coefficients[,1] != 0,])


# Prepare the data
y_test_pred <- predict(opt_lasso_model,
                       newx = as.matrix(test %>% select(Channel1:Channel100)),
                       s = optimal_lambda)

# Create a scatter plot of original test values versus predicted test values
plot(test$Fat, y_test_pred, main = "Scatter Plot: Original vs. Predicted (LASSO)",
     xlab = "Original Test Values", ylab = "Predicted Test Values")
abline(0, 1, col = "red")




