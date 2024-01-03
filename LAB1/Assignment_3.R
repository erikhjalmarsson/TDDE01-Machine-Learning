# -------------------------------------------
# Title:  Logistic regression 
# Author: Erik Hjalmarsson
# File:   Assignment_3.R
# Date:   2023-12-06
# Last    modified: 2023-12-13
# -------------------------------------------

# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)


#--------------------------------------------
# Functions
#--------------------------------------------

# Misclassification rate
misclass <- function(x, x1){
  n <- length(x)
  return(1 - sum(diag(table(x, x1))) / n)
}

#--------------------------------------------
# Task 1
#--------------------------------------------

data <- read.csv("pima-indians-diabetes.csv", header = FALSE)

pgc       <- data$V2
dpf       <- data$V7
age       <- data$V8
diabetes  <- data$V9

ggplot(data, 
       aes(x = age, y = pgc, color = dpf)) +
       geom_point() +
       labs(x = "Age", y = "Plasma Glucose Concentration", color = "Diabetes") +
       scale_color_gradient(low = "blue", high = "red")

#--------------------------------------------
# Task 2
#--------------------------------------------

logistic_model <- glm(diabetes ~ pgc + age, 
                      data=data,
                      family= 'binomial')

#Type="Response" gives the predicted probabilities of the form P(Y=1|X)
probabilities <- predict(logistic_model,
                         data=data,
                         type="response")
r <- 0.5
predictions <- ifelse(probabilities > r, 1, 0)

#Retrieving the probabilistic equation
coef <- logistic_model$coefficients
cat("Probabilistic equation = 1 / (1 + exp(-(", coef[1], 
    " + ", coef[2], " * x1 + ", coef[3], " * x2)))")

confusion_matrix <- table(predictions, diabetes)
print(confusion_matrix)

misclassrate <- misclass(predictions, data$V9)
print(paste("misclassrate: ", misclassrate))

print(ggplot(data, aes(x = age, y = pgc, color = predictions)) +
        geom_point() +
        labs(x = "Age",
             y = "Plasma Glucose Concentration",
             color = "Diabetes") +
        scale_color_gradient(low = "blue", high = "red"))

#--------------------------------------------
# Task 3
#--------------------------------------------
# Calculate the slope and intercept of the decision boundary
intercept <- -coef[1] / coef[["pgc"]]
slope <- -coef[["age"]] / coef[["pgc"]]

# Add the decision boundary to the plot
print(ggplot(data, aes(x = age, y = pgc, color = predictions)) +
             geom_point() +
             geom_abline(intercept = intercept,
                         slope = slope,
                         color = "black",
                         linetype = "dashed") +
             labs(x = "Age",
                  y = "Plasma Glucose Concentration",
                  color = "Diabetes") +
             scale_color_gradient(low = "blue", high = "red"))

#--------------------------------------------
# Task 4
#--------------------------------------------

# Using thresholds r= 0.2 & r = 0.8

# r = 0.2
r <- 0.2
pred_r_0.2 <- ifelse(probabilities > r, 1, 0)

ggplot(data, aes(x = age, y = pgc, color = pred_r_0.2)) +
       geom_point() +
       labs(x = "Age, r = 0.2",
            y = "Plasma Glucose Concentration",
            color = "Diabetes") +
       scale_color_gradient(low = "blue", high = "red")

r <- 0.8

pred_r_0.8 <- ifelse(probabilities > r, 1, 0)
print(ggplot(data, aes(x = age, y = pgc, color = pred_r_0.8)) +
             geom_point() +
             labs(x = "Age, r = 0.8",
                  y = "Plasma Glucose Concentration",
                  color = "Diabetes") +
             scale_color_gradient(low = "blue", high = "red"))

#--------------------------------------------
# Task 5
#--------------------------------------------

z1 <- pgc^4
z2 <- (pgc^3)*age
z3 <- (pgc^2)*(age^2)
z4 <- (pgc)*(age^3)
z5 <- (age^4)

improved_logistic_model <- glm(diabetes ~ pgc + age + z1 + z2 + z3 + z4 + z5,
                               data=data,
                               family= 'binomial')

new_probability <- predict(improved_logistic_model,
                           data=data + z1 + z2 + z3 + z4 + z5,
                           type="response")
r <- 0.5
new_pred <- ifelse(new_probability > r, 1, 0)


ggplot(data + z1 + z2 + z3 + z4 + z5,
       aes(x = age, y = pgc, color = new_pred)) +
       geom_point() +
       labs(x = "Age", y = "Plasma Glucose Concentration",
            color = "Diabetes") +
       scale_color_gradient(low = "blue", high = "red")

misclassrate <- misclass(new_pred, diabetes)
print(paste("misclassrate: ", misclassrate))

confusion_mtrx <- table(new_pred, diabetes)
print(confusion_mtrx)


existing_plot <- ggplot(data + z1 + z2 + z3 + z4 + z5,
                        aes(x = age, y = pgc, color = new_pred)) +
                        geom_point() +
                        labs(x = "Age", y = "Plasma Glucose Concentration", color = "Diabetes") +
                        scale_color_gradient(low = "blue", high = "red")

# Create contour data
x_vals <- seq(min(age), max(age), length.out = 100)
y_vals <- seq(min(pgc), max(pgc), length.out = 100)

new_data <- expand.grid(age = x_vals, pgc = y_vals)
new_data$z1 <- new_data$pgc^4
new_data$z2 <- new_data$pgc^3 * new_data$age
new_data$z3 <- new_data$pgc^2 * new_data$age^2
new_data$z4 <- new_data$pgc * new_data$age^3
new_data$z5 <- new_data$age^4

# Add the decision boundary to the contour data
new_data$decision_boundary <- ifelse(predict(improved_logistic_model, newdata = new_data, type = "response") > 0.5, 1, 0)

# Add contour plot to existing plot
combined_plot <- existing_plot +
  geom_contour(data = new_data, aes(x = age, y = pgc, z = decision_boundary), color = "red", linetype = 8)

# Print the combined plot
print(combined_plot)
