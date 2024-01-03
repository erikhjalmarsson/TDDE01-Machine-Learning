# -------------------------------------------
# Title:  Principal Component Analysis
# Author: Erik Hjalmarsson
# File:   Assignment3.R
# Date:   2024-01-03
# Last modified: 2024-01-03
# -------------------------------------------

# Load necessary libraries
library(caret)
library(ggplot2)
library(dplyr)
library(tidyr)

imported_data <- read.csv("communities.csv", header = TRUE)

#--------------------------------------------
# Task 1
#--------------------------------------------

#Scaling data with preprocess using center
data<- imported_data%>%select(-ViolentCrimesPerPop)
scaler <- preProcess(data);
dataS <- predict(scaler, data)

#Calculating the eigenvalues
cov_matrix <- cov(dataS)
pca <- eigen(cov_matrix)
eigenvalues <- pca$values

#Calculatng variance
variance <- eigenvalues / sum(eigenvalues)

#Calculating number of PCs required for variance of 0.95
cumulative_variance <- cumsum(variance)
targetPCs <- which(cumulative_variance >= 0.95)
targetPC <- targetPCs[1]

#printing response
print(paste("Numbers of PCs needed for variance of 95%: ", targetPC))
print(paste("PC1:", eigenvalues[1], "%, PC2:", eigenvalues[2], "%"))


#--------------------------------------------
# Task 2
#--------------------------------------------

pcs_princomp <- princomp(dataS)

#Plotting trace plot of pc1

print(ggplot(as.data.frame(pcs_princomp$loadings[,1]), aes(x= seq(from = 1, to = 100, by = 1), y = pcs_princomp$loadings[,1])) + geom_point(alpha = 0.7) +  labs(title = "Trace plot of PC1", x = "Index", y = "Principal Component 1"))

#Which 5 features contribute the most
pc1 <- pcs_princomp$loadings[,1]
pc1_abs <- abs(pc1)
pc1_sorted <- sort(pc1_abs, decreasing = TRUE)
contributers <- head(pc1_sorted, n=5)

print("5 most contributers to PC1 by absolute value:")
print(contributers)


print(ggplot(data.frame(pc_1 = pcs_princomp$scores[,1],
                        pc_2 = pcs_princomp$scores[,2]),
                        aes(x = pc_1, y = pc_2,
                            color = imported_data$ViolentCrimesPerPop)) 
                        + geom_point(alpha = 0.7) 
                        + scale_color_gradient(name = "ViolentCrimesPerPop",
                                               low = "blue", high = "red") 
                        + labs(title = "Scatter Plot of PC1 vs PC2",
                               x = "Principal Component 1",
                               y = "Principal Component 2"))


#--------------------------------------------
# Task 3
#--------------------------------------------

# partitioning the data
n <- dim(imported_data)[1]
set.seed(12345)
id <- sample(1:n, floor(n * 0.5))
train <- imported_data[id, ]
test <- imported_data[-id,]

# Scaling the data
scaler <- preProcess(train)
trainS <- predict(scaler, train)
testS <- predict(scaler, test)

# Training linnear regression model and making predictions
lr_model <- lm(trainS$ViolentCrimesPerPop ~. , data = trainS)
training_hat <- predict(lr_model, newdata = trainS, type= "response")
test_hat <- predict(lr_model, newdata = testS, type= "response")

# Calculating Mean Squared Error of test and train data
MSE_test <- mean((testS$ViolentCrimesPerPop-test_hat)^2)
MSE_train <- mean((trainS$ViolentCrimesPerPop-training_hat)^2)


#Printing answer
print(paste("Error for lr training data: ", MSE_train))
print(paste("Error for lr test data: ", MSE_test))

#--------------------------------------------
# Task 4
#--------------------------------------------

#Calculating matrices for use in cost function
x_train <- as.matrix(trainS%>%select(-ViolentCrimesPerPop))
x_test <- as.matrix(testS%>%select(-ViolentCrimesPerPop))
y_train <- trainS$ViolentCrimesPerPop
y_test <- testS$ViolentCrimesPerPop

#Defining parameters to store errors for each iteration
MSE_test_it <- NULL
MSE_train_it <- NULL

#cost function of linear regression dependent on theta
cost <- function(theta) {
  MSE_test_temp <- mean((y_test - x_test %*% theta)^2)
  MSE_train_temp <- mean((y_train - x_train %*% theta)^2)
  
  MSE_test_it <<- c(MSE_test_it, MSE_test_temp)
  MSE_train_it <<- c(MSE_train_it, MSE_train_temp)
  
  return (MSE_train_temp)
}

#Optimizing theta
theta <- rep(0,100)
optimal_theta <- optim(theta, cost, method = "BFGS")

#Plotting the errors
plot(MSE_test_it,
     ylim = c(0.2, 0.8), ylab = "Mean square error",
     xlab = "# of calculating",
     pch = ".", xlim = c(500, 20000),
     col = "red")
points(MSE_train_it, col = "blue", pch = ".")
legend("topright", c("Test Data", "Training Data"), col = c("red", "blue"), lwd = 1)

#Iteration where optimal test error is reached
optIt <- which(MSE_train_it == min(MSE_train_it))

#Printing the optimal test errors:
print(paste("Optimal training error", MSE_train_it[optIt], "reached in calculation", optIt[1]))
print(paste("Corresponding test error",MSE_test_it[optIt], "reached in calculation", optIt[2]))

