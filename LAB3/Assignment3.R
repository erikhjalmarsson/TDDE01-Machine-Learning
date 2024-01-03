# -------------------------------------------
# Title: Neural Networks
# Author: Erik Hjalmarsson
# File: Assignment3.R
# Date: 2024-01-03
# Last modified: 2024-01-03
# -------------------------------------------

library(neuralnet)



#--------------------------------------------
# Task 1
#--------------------------------------------
set.seed(1234567890)

Var <- runif(500, 0, 10)
mydata <- data.frame(Var, Sin = sin(Var))
tr <- mydata[1:25,]  # Training
te <- mydata[26:500,]  # Test

# Random initialization of the weights in the interval [-1, 1]
set.seed(1234567890)
winit <- runif(10, -1, 1) 
# Standard activation function is sigmoid
nn <- neuralnet(Sin ~ Var, data = tr, hidden = c(10), linear.output = TRUE, startweights = winit)

# Plot of the training data (black), test data (blue), and predictions (red)
plot(tr, cex = 2)
points(te, col = "blue", cex = 1)
points(te[,1], predict(nn, te), col = "red", cex = 1)

#--------------------------------------------
# Task 2
#--------------------------------------------

# Define custom activation functions
custom_linear <- function(x) x
custom_softplus <- function(x) log(1 + exp(x))
custom_relu <- function(x) ifelse(x > 0, x, 0)

#Linear activation function:
nn_linear <- neuralnet(Sin ~ Var, data = tr, hidden = c(10), linear.output = TRUE, act.fct = custom_linear, startweights = winit)

#ReLU (Rectified Linear Unit) Activation:
nn_relu <- neuralnet(Sin ~ Var, data = tr, hidden = c(10), linear.output = TRUE, act.fct = custom_relu, startweights = winit)

#Softplus Activation:
nn_softplus <- neuralnet(Sin ~ Var, data = tr, hidden = c(10), linear.output = TRUE, act.fct = custom_softplus, startweights = winit)

# Plot of the training data (black), test data (blue), and predictions (red)
plot(tr, cex = 2)
points(te, col = "blue", cex = 1)
points(te[,1], predict(nn_linear, te), col = "red", cex = 1)

# Plot of the training data (black), test data (blue), and predictions (red)
plot(tr, cex = 2)
points(te, col = "blue", cex = 1)
points(te[,1], predict(nn_relu, te), col = "red", cex = 1)

# Plot of the training data (black), test data (blue), and predictions (red)
plot(tr, cex = 2)
points(te, col = "blue", cex = 1)
points(te[,1], predict(nn_softplus, te), col = "red", cex = 1)

#--------------------------------------------
# Task 3
#--------------------------------------------

# Generate 500 random points in the interval [0, 50]
set.seed(1234567890)
new_points <- runif(500, 0, 50)

# Create a data frame with the new points and their sine values
new_data <- data.frame(new_points, Sin = sin(new_points))

# Use the neural network to predict the sine function values for the new points
predicted_values <- predict(nn, new_data[1])

# Visualize performance of predictions with nn learned in task 1
plot(new_data, cex = 2, xlab = "Var", ylab = "Sin", ylim = c(-10, 2))
points(new_data[, 1], predicted_values, col = "blue", cex = 0.8)


#--------------------------------------------
# Task 4
#--------------------------------------------

nn$weights

#--------------------------------------------
# Task 5
#--------------------------------------------

# Generate 500 random points in the interval [0, 10]
set.seed(1234567890)
new_points2 <- runif(500, 0, 10)

new_data2 <- data.frame(new_points2, Sin = sin(new_points2))

train <- new_data2[1:500, ]

nn2 <- neuralnet(new_points2 ~ ., data = train, hidden = c(10), threshold = 0.1, linear.output = TRUE, startweights = winit)

# Plot of the train data (black) and predictions (red)
plot(train[, 2], train[, 1], cex = 2, xlab = "Sin", ylab = "x")
points(train[, 2], predict(nn2, train), col = "red", cex = 0.8)


