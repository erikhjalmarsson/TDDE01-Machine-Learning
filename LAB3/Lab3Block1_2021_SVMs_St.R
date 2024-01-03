# Lab 3 block 1 of 732A99/TDDE01/732A68 Machine Learning
# Author: jose.m.pena@liu.se
# Made for teaching purposes

library(kernlab)
set.seed(1234567890)

data(spam)
foo <- sample(nrow(spam))
spam <- spam[foo,]

scaler <- preProcess(spam[1:3000,-58])
tr <- predict(scaler, spam[1:3000, ])
trva <- predict(scaler, spam[1:3800, ])
va <- predict(scaler, spam[3001:3800, ])
te <- predict(scaler, spam[3801:4601, ])

# Never scale the whole dataset before partitioning
#spam[,-58]<-scale(spam[,-58])
#tr <- spam[1:3000, ]
#va <- spam[3001:3800, ]
#trva <- spam[1:3800, ]
#te <- spam[3801:4601, ] 

by <- 0.3
err_va <- NULL
for(i in seq(by,5,by)){
  filter <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=i,scaled=FALSE)
  mailtype <- predict(filter,va[,-58])
  t <- table(mailtype,va[,58])
  err_va <-c(err_va,(t[1,2]+t[2,1])/sum(t))
}

filter0 <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter0,va[,-58])
t <- table(mailtype,va[,58])
err0 <- (t[1,2]+t[2,1])/sum(t)
err0

filter1 <- ksvm(type~.,data=tr,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter1,te[,-58])
t <- table(mailtype,te[,58])
err1 <- (t[1,2]+t[2,1])/sum(t)
err1

filter2 <- ksvm(type~.,data=trva,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter2,te[,-58])
t <- table(mailtype,te[,58])
err2 <- (t[1,2]+t[2,1])/sum(t)
err2

filter3 <- ksvm(type~.,data=spam,kernel="rbfdot",kpar=list(sigma=0.05),C=which.min(err_va)*by,scaled=FALSE)
mailtype <- predict(filter3,te[,-58])
t <- table(mailtype,te[,58])
err3 <- (t[1,2]+t[2,1])/sum(t)
err3

# Questions

# 1. Which filter do we return to the user ? filter0, filter1, filter2 or filter3? Why?

# 2. What is the estimate of the generalization error of the filter returned to the user? err0, err1, err2 or err3? Why?

# 3. Implementation of SVM predictions.

sv<-alphaindex(filter3)[[1]]
co<-coef(filter3)[[1]]
inte<- - b(filter3)
k<-NULL

# Create an RBF kernel function with sigma = 0.05
rbf_kernel <- rbfdot(sigma = 0.05)

for(i in 1:10){ # We produce predictions for just the first 10 points in the dataset.
  k2<-NULL
  for(j in 1:length(sv)){
    # Calculate the kernel function value between the test point and the support vector
    kernel_value <- rbf_kernel(as.numeric(spam[sv[j], -58]), as.numeric(spam[i, -58]))
    
    # Multiply the kernel function value by the corresponding coefficient
    weighted_kernel_value <- co[j] * kernel_value
    
    # Add the weighted kernel value to the decision function value
    k2 <- c(k2, weighted_kernel_value)
  }
  k<-c(k, sum(k2) + inte)# Your code here)
}
k
predict(filter3,spam[1:10,-58], type = "decision")

