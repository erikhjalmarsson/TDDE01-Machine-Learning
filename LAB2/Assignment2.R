# -------------------------------------------
# Title:  Decision Trees
# Author: Erik Hjalmarsson
# File:   Assignment2.R
# Date:   2023-12-06
# Last modified: 2023-12-31
# -------------------------------------------

# Load necessary libraries
library(dplyr)
library(tidyr)
library(tree)

# csv2 since it is semicolon separated file
data_import <- read.csv2("bank-full.csv", stringsAsFactors = T)

#--------------------------------------------
# Functions
#--------------------------------------------

misclass <- function(x){
  return(x[1]/x[2])
}

misclass_table <- function(x, x1){
  n <- length(x)
  return(1 - sum(diag(table(x, x1))) / n)
}


#--------------------------------------------
# Task 1 - Partitioning the data
#--------------------------------------------

# Removing "duration" from dataset
data <- data_import %>% select(-duration)

# Partitioning the data into train, val and test

# Train
n = dim(data)[1]
set.seed(12345)
id = sample(1:n, floor(n*0.4))
train = data[id,]

# Validation
id1 = setdiff(1:n, id)
set.seed(12345)
id2 = sample(id1, floor(n*0.3))
valid = data[id2, ]

# Test
id3 = setdiff(id1, id2)
test = data[id3,]

#--------------------------------------------
# Task 2 - Setting up the decision trees & calculating misclassificationrate
#--------------------------------------------

#####
# Trees

tree_a  = tree(y~., data = train)
tree_b  = tree(y~., data = train, control = tree.control(nrow(train), minsize = 7000))
tree_c  = tree(y~., data = train, control = tree.control(nrow(train), mindev = 0.0005))

t_a <- summary(tree_a)
t_b <- summary(tree_b)
t_c <- summary(tree_c)

MCR_a <- misclass(t_a$misclass)
MCR_b <- misclass(t_b$misclass)
MCR_c <- misclass(t_c$misclass)


p2_a <- predict(tree_a, newdata=valid, type="class")
p2_b <- predict(tree_b, newdata=valid, type="class")
p2_c <- predict(tree_c, newdata=valid, type="class")

MCR2_a <- mean(p2_a != valid$y)
MCR2_b <- mean(p2_b != valid$y)
MCR2_c <- mean(p2_c != valid$y)

cat("---------- Tree A ----------")
cat("\nMisclassification rate for tree_a train: ", MCR_a)
cat("\nMisclassification rate for tree_a validation: ", MCR2_a)

cat("\n---------- Tree B ----------")
cat("\nMisclassification rate for tree_b train: ", MCR_b)
cat("\nMisclassification rate for tree_b validation: ", MCR2_b)

cat("\n---------- Tree C ----------")
cat("\nMisclassification rate for tree_c train: ", MCR_c)
cat("\nMisclassification rate for tree_c validation: ", MCR2_c)

cat("\n------ Terminal Nodes ------")
cat("\nTerminal nodes tree_a: ", t_a$size)
cat("\nTerminal nodes tree_b: ", t_b$size)
cat("\nTerminal nodes tree_c: ", t_c$size)


#--------------------------------------------
# Task 3
#--------------------------------------------

trainScore = rep(0,50)
testScore = rep(0,50)

set.seed(12345)
for(i in 2:50){
  prunedTree = prune.tree(tree_c, best = i)
  pred = predict(prunedTree, newdata = valid, type="tree")
  trainScore[i] = deviance(prunedTree)/count(train) # Divide by the size of the dataset?
  testScore[i] = deviance(pred)/count(valid)
}

plot(2:50, 
     trainScore[2:50],
     type="b",
     col="red",
     ylim=c(min(as.numeric(trainScore[-1]), as.numeric(testScore[-1])),
            max(as.numeric(trainScore), as.numeric(testScore))),
     xlab = "Amount of leaves",
     ylab = "Deviance")

points(2:50, testScore[2:50], type="b", col="blue")
legend("topright", legend = c("trainScore", "testScore"), col = c("red", "blue"), pch = 16)

# Bias - Variance Tradeoff
# When creating a good model it is necessary to think about the bias - variance tradeoff.
# Bias - variance tradeoff constitutes the balance between model complexity and model flexibility. 
# Low bias is connected to a complex model leading in turn to high variance, meaning the model finds it difficult to adjust to new data. 
# Therefore, it is important to optimize the balance the bias and variation to find the optimal model. 

print(min(as.numeric(testScore[2:50]))) # 0.604771
print(which.min(testScore[2:50]) +1) # Have to add one to the index since we start on 2

# By picking the smallest value of the testScore vector we get the optimal amount of leaves to be 22

optimal_tree = prune.tree(tree_c, best = 22)
fit <- predict(optimal_tree, newdata = valid, type = "class")
table(true = valid$y, predicted = fit)
print(paste("Missclassrate for 22 leaves: ", misclass_table(valid$y, fit)))
# This gives 88% accuracy which is bad since labeling all to no gives accuracy of 87%

plot(optimal_tree)
text(optimal_tree, pretty=0)

# When observing the final tree, it is evident that pdays and poutcome is one of the most important variables since 
# all classifications leading to yes are based on those variables

#--------------------------------------------
# Task 4
#--------------------------------------------

# Confusion Matrix
print("Confusion matrix for the test data")
testFit <- predict(optimal_tree, newdata = test, type="class")
table(true = test$y, predicted = testFit)

# Accuracy: 0.891035092892952
cm <- table(test$y, testFit)
accuracy <- sum(diag(cm)) / sum(cm)
print(paste("Accuracy:", accuracy))

# F1 - Score: 0.224554039874082
cm <- as.vector(table(test$y, testFit))
recall <- cm[4]/(cm[2]+cm[4]) # TP/P
precision <- cm[4]/(cm[4]+cm[3]) #TP/(TP +FP)

F1 <- (2*precision*recall)/(precision + recall)
print(paste("F1 score: ",F1))

#--------------------------------------------
# Task 5
#--------------------------------------------

loss_matrix <- matrix(c(0, 1, 5, 0), byrow=TRUE, nrow=2)
print(loss_matrix)

prob <- predict(optimal_tree, newdata = test)

losses <- prob %*% loss_matrix

res <- apply(losses, MARGIN = 1, FUN = which.min)
prediction <- levels(test$y)[res]
cm2 <- table(prediction, test$y)
cm2

accuracy <- sum(diag(cm2)) / sum(cm2)
print(paste("Accuracy:", accuracy))

# F1 - Score
cm <- as.vector(table(prediction, test$y))
recall <- cm[4]/(cm[3]+cm[4]) # TP/P
precision <- cm[4]/(cm[2]+cm[4]) #TP/(TP +FP)

F1 <- (2*precision*recall)/(precision + recall)
print(paste("F1 score: ",F1))

#--------------------------------------------
# Task 6
#--------------------------------------------

probabilities <- predict(optimal_tree, newdata = test, type="vector")
predicted_classes <- vector("character", length = nrow(probabilities))

tpr <- numeric()
fpr <- numeric()
pre <- numeric()
rec <- numeric()

for(i in seq(0.05, 0.95, by = 0.05 )){
  
  change <- ifelse(probabilities[,2] > i, "yes", "no")
  changetable <- table(predicted = change, true = test$y)
  
  TP <- changetable[4]
  FP <- changetable[2]
  P <- changetable[3]+changetable[4]
  N <- changetable[1]+changetable[2]
  tpr <- c(tpr, (TP/P))
  fpr <- c(fpr, (FP/N))
  pre <- c(pre, TP/(TP+FP))
  rec <- c(rec, TP/(P))
  
}
plot(fpr, tpr,xlab ="False Positive Rate", ylab ="True Positive Rate", pch = 16, type = "b")


#####
# Regression model

reg_model <- glm(y ~.,data = train, family="binomial")
probabilities_reg <- predict(reg_model, newdata = test, type="response")

tpr2 <- numeric()
fpr2 <- numeric()
pre2 <- numeric()
rec2 <- numeric()

for(i in seq(0.05, 0.95, by = 0.05 )){
  
  change <- ifelse(probabilities_reg > i, "yes", "no")
  changetable <- table(predicted = change, true = test$y)
  
  TP <- changetable[4]
  FP <- changetable[2]
  P <- changetable[3]+changetable[4]
  N <- changetable[1]+changetable[2]
  tpr2 <- c(tpr2, (TP/P))
  fpr2 <- c(fpr2, (FP/N))
  pre2 <- c(pre2, TP/(TP+FP))
  rec2 <- c(rec2, TP/(P))
  
}

points(fpr2, tpr2, pch = 16, type = "b", col="red")
legend("topleft", legend = c("Decision Tree", "Logistic Regression"), col = c("black", "red"), pch = 16)

plot(rec, pre, pch = 16, type = "b", xlab ="Recall", ylab ="Precision")
points(rec2, pre2, pch = 16, type = "b", col="red")
legend("topright", legend = c("Decision Tree", "Logistic Regression"), col = c("black", "red"), pch = 16)



