# -------------------------------------------
# Title: . Handwritten digit recognition with 
#          K nearest neighbors (KNN).
# Author: Erik Hjalmarsson
# File: Assignment1.R
# Date: 2023-12-06
# Last modified: 2023-12-13
# -------------------------------------------

# Load necessary libraries
library(kknn)
library(dplyr)
library(tidyr)

#--------------------------------------------
# Task 1
#--------------------------------------------

set.seed(12345)

#Reading dataframe from csv file
data <- read.csv("optdigits.csv", header = FALSE)

#Splitting data into train, validation and test
n <- dim(data)[1]
id <- sample(1:n, floor(n * 0.5))
train <- data[id, ]

id1 <- setdiff(1:n, id)
id2 <- sample(id1, floor(n * 0.25))
validation <- data[id2, ]

id3 <- setdiff(id1, id2)
test <- data[id3, ]

#--------------------------------------------
# Task 2
#--------------------------------------------

#Our kknn model on train data
test_model <- kknn(as.factor(V65) ~ ., train, test, k = 30, kernel = "rectangular")
train_model <- kknn(as.factor(V65) ~ ., train, train, k = 30, kernel = "rectangular")

#Uses kknn_model to predict values of test dataset
test_prediction <- predict(test_model)
train_prediction <- predict(train_model)

#Confusion matrix for result of test predictions
test_conf_matrix <- table(test$V65, test_prediction)
train_conf_matrix <- table(train$V65, train_prediction)

#Prints confusion matrixes
print(test_conf_matrix)
print(train_conf_matrix)

#Function for missclassification
missclass <- function(x, x1){
  n <- length(x)
  return(1 - sum(diag(table(x, x1))) / n)
}

test_missclass <- missclass(test$V65, test_prediction)
train_missclass <- missclass(train$V65, train_prediction)


print(paste("Test Misclassification Error: ", test_missclass))
print(paste("Train Misclassification Error: ", train_missclass))

#--------------------------------------------
# Task 3
#--------------------------------------------

train_8 <- train[train$V65 == "8",]
prob_model <- kknn(as.factor(V65)~., train, train_8, k = 30, kernel = "rectangular")
probabilities <- predict(prob_model, newdata = train_8, type = "prob")
prob_8 <- probabilities[, "8"]
indices <- row.names(train_8)

# Combine the indices and probabilities into a data frame
result <- data.frame(indices = indices, Prob_8 = prob_8)

# Sort the data frame by the probabilities
result <- result %>% arrange(result$Prob_8, decreasing = TRUE)

# Print the first 10 rows of the result
easiest <- tail(result, 2)
hardest <- head(result, 3)

#create a function shell
plot_digit <- function(case_index){
  case_features <- train[case_index, -ncol(train)]
  
  # Reshape the features into an 8x8 matrix
  matrix_features <- matrix(as.numeric(case_features), byrow = TRUE, ncol = 8)
  
  # Create the heatmap
  heatmap(matrix_features, Colv = NA, Rowv = NA)
}

plot_digit(easiest[1, 1])
plot_digit(easiest[2, 1])
plot_digit(hardest[3, 1])
plot_digit(hardest[2, 1])
plot_digit(hardest[3, 1])

#--------------------------------------------
# Task 4
#--------------------------------------------

# Initialize the vectors to store the errors
train_errors <- numeric(30)
validation_errors <- numeric(30)
validation_cross_entropy <- numeric(30)

# Loop over the values of K
for (k_value in 1:30) {
  # Fit a K-nearest neighbor classifier to the training data
  train_model_k <- kknn(as.factor(V65) ~ ., train = train, test = train, k = k_value, kernel = "rectangular")
  validation_model_k <- kknn(as.factor(V65) ~ ., train = train, test = validation, k = k_value, kernel = "rectangular")
  
  # Predict the classes for the training and validation data
  train_predictions_k <- predict(train_model_k)
  validation_predictions_k <- predict(validation_model_k)

  # Compute the missclassification errors
  train_errors[k_value] <- missclass(train$V65, train_predictions_k)
  validation_errors[k_value] <- missclass(validation$V65, validation_predictions_k)

  # Compute the cross entropy of validation predictions
  for (y in 0:9) {
    indicatorY <- validation_model_k$prob[which(validation$V65 == y), y+1]
    logProbY <- log(indicatorY + 1e-15)
    validation_cross_entropy[k_value] <- validation_cross_entropy[k_value] - sum(logProbY)
  }
}

# Plot the errors against the values of K
plot(1:30,
     train_errors,
     type = "l",
     col = "blue",
     xlab = "K",
     ylab = "Missclassification error")

lines(1:30, 
      validation_errors, 
      col = "red")

legend("bottomright",
       legend = c("Training", "Validation"),
       fill = c("blue", "red"))

k_optimal <- which.min(validation_errors)

optimal_test_model <- kknn(as.factor(V65) ~ ., train = train, test = test, k = k_optimal, kernel = "rectangular")
optimal_test_predictions <- predict(optimal_test_model, newdata = test)
optimal_test_missclass <- missclass(test$V65, optimal_test_predictions)


k_optimal_entropy <- which.min(validation_cross_entropy)

#plotting cross-entropy against K
plot(1:30, validation_cross_entropy, type = "l", col = "blue", xlab = "K", ylab = "Cross-entropy")

#Printing key values
print(paste("Optimal K for missclassification rate: ", k_optimal))
print(paste("Optimal Training Misclassification Error: ", train_errors[k_optimal]))
print(paste("Optimal Validation Misclassification Error: ", validation_errors[k_optimal]))
print(paste("Optimal Test Misclassification Error: ", optimal_test_missclass))


print(paste("Optimal K for cross-entropy: ", k_optimal_entropy))
