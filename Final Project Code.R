# Final Project Code - DS740 Fall '18
# By Deane Buckingham

# ********************** Reading in and prepping the data **********************************

first.data <- read.csv("/Users/deanebuckingham/Documents/DS740 - Data Mining/Final Project/avocado.csv")

# Creating an Income column to potentially use in analysis
first.data$Income <- first.data$AveragePrice * first.data$Total.Volume
# keeping output from printing in scientific notation.
options(scipen = 999)

# Checking for blanks or NAs.
n <- nrow(first.data)
p <- ncol(first.data)
rownames <- c("# of NA", "# of blanks", "ratio of NA", "ratio of blanks")
colnames <- colnames(first.data)
first_look <- matrix(nrow=4, ncol=p, dimnames = list(rownames, colnames))
for (i in 1:p){
  first_look[1,i] <- sum(is.na(first.data[,i]))
  first_look[2,i] <- length(which(first.data[,i] == " "))
  first_look[3,i] <- first_look[1,i]/n
  first_look[4,i] <- first_look[2,i]/n
}
View(first_look)
# No blanks or NAs at all.

# Making sure the variables that need to be factors... are so.
first.data$X <- as.factor(first.data$X)
first.data$year <- as.factor(first.data$year)

# Converting the Date column to the proper format.
library(lubridate)
first.data$Date <- ymd(first.data$Date)
head(first.data$Date)

# ******************* Linear Regression on logTotal.Volume ************************

# AveragePrice, Income and Total.Volume were all right-skewed a little. They looked better 
# after a log transformation.
first.data$logTotal.Volume <- log(first.data$Total.Volume)
first.data$logIncome <- log(first.data$Income)
first.data$logAveragePrice <- log(first.data$AveragePrice)

# We'll try a large Linear Regression on logTotal.Volume first.
full.fit <- lm(logTotal.Volume ~ . -Total.Volume -X -Date -region -logIncome -logAveragePrice, 
               data=first.data)
        # I left out a few necessary columns.
summary(full.fit)
hist(full.fit$residuals) # Residuals approx normal.
boxplot(full.fit$residuals)

# Check for collinearity. 
# Try to find the VIF on my model
library(car)
vif(full.fit) # I found out from this function that the PLU and Bag columns are 
      # increadibly collinear since their scores are well above the threshold of 10. 

# I will use AIC to select the predictors.
library(leaps)
sub.full <- regsubsets(logTotal.Volume ~ .-Total.Volume -X -Date -logIncome -logAveragePrice, 
                       data=first.data, nvmax=13, really.big=TRUE)
plot(sub.full)
sub.full.summary <- summary(sub.full)
plot(sub.full.summary$bic, xlab="# of Variables", ylab = "BIC", type="l", lwd=2)

# a smaller lm on logTotal.Volume (type & region) using training and testing datasets.
set.seed(394)
n <- nrow(first.data) 
test.data <- sample(n, size=(.2*n))

smaller.fit <- lm(logTotal.Volume ~ type + region, data=first.data[-test.data, ])
summary(smaller.fit)
hist(smaller.fit$residuals) # Residuals approx normal.
boxplot(smaller.fit$residuals)
# Adj R-squared:  0.9583

pred.vol <- predict(smaller.fit, newdata=first.data[test.data, ])
mean((first.data$logTotal.Volume[test.data] - pred.vol)^2) # about 0.22.


# ****************** making logAveragePrice as my response variable ********************

# A large Linear Regression model on logAveragePrice.
full.fit <- lm(logAveragePrice ~ . -AveragePrice -logIncome -logTotal.Volume -X -Date, 
               data=first.data)
summary(full.fit)
hist(full.fit$residuals) # residuals appear approx normal.
boxplot(full.fit$residuals)
# Adj R-squared:  0.6144

# Checking for collinearity. 
# Try to find the VIF on my model
library(car)
vif(full.fit) # all of the PLU and Bag columns are terribly collinear again, as is Income.

# I will use AIC to select the predictors.
library(leaps)
sub.full <- regsubsets(logAveragePrice ~ .-AveragePrice -logIncome -logTotal.Volume -X -Date
                       , data=first.data, nvmax=13, really.big = TRUE)
sub.full.summary <- summary(sub.full)
plot(sub.full)
sub.full.summary$bic
plot(sub.full.summary$bic, xlab="# of Variables", ylab = "BIC", type="l", lwd=2)
# Looks like 5 variables is best.

# applying the smaller lm on the training and testing datasets.
set.seed(394)
n <- nrow(first.data) 
test.data <- sample(n, size=(.2*n))

smaller.fit <- lm(logAveragePrice ~ type + year + X4046 + Total.Bags + region, 
                  data=first.data[-test.data, ])
summary(smaller.fit)
hist(smaller.fit$residuals)
boxplot(smaller.fit$residuals)
plot(smaller.fit)

pred.vol <- predict(smaller.fit, newdata=first.data[test.data, ])
mean((first.data$logAveragePrice[test.data] - pred.vol)^2)

# ****************************** Using Lesson 5 Techniques **********************************

# The below code is using AveragePrice as the response. I ran the same code earlier
# with Total.Volume as the response.

#labels
x = model.matrix(AveragePrice ~ .-X-Date-logIncome-logTotal.Volume-region-logAveragePrice ,
                 data=first.data)[,-1] 
y = first.data[,3]

#fit Ridge Regression using most predictors
library(glmnet) 
lambdalist = exp((1200:-1200)/100)  # must be in order large to small
#fit ridge regression - need alpha = 0
RRfit = glmnet(x, y, alpha = 0,lambda=lambdalist)
coef(RRfit,s=-2) # we see the coefficients for s=0.1
plot(RRfit, xvar="lambda",xlim=c(-5,8))

#fit LASSO - need alpha =1
LASSOfit = glmnet(x, y, alpha = 1,lambda=lambdalist)
coef(LASSOfit,s=-3) 
plot(LASSOfit, xvar="lambda")

#fit ENET
ENETfit = glmnet(x, y, alpha = 0.75,lambda=lambdalist)
coef(ENETfit,s=-3)
plot(ENETfit, xvar="lambda")

# INCORPORATING CV
set.seed(394)
ncv = 10
n <- nrow(first.data)
groups=c(rep(1:ncv,floor(n/ncv)),(1:(n-ncv*floor(n/ncv))))
cvgroups = sample(groups,n)
# lambdalist = exp((1200:-1200)/100)  # as from before...order large to small

#RR cross-validation
cvRRglm = cv.glmnet(x, y, lambda=lambdalist, alpha = 0, nfolds=ncv, foldid=cvgroups)
plot(cvRRglm$lambda,cvRRglm$cvm,type="l",lwd=2,col="red",xlab="lambda",ylab="CV(10)",
     xlim=c(0,2),ylim = c(0,0.2)) #looking at only a narrow window of the graph.

whichlowestcvRR = order(cvRRglm$cvm)[1]; min(cvRRglm$cvm)
bestlambdaRR = lambdalist[whichlowestcvRR]; bestlambdaRR
abline(v=bestlambdaRR)

#LASSO cross-validation
cvLASSOglm = cv.glmnet(x, y, lambda=lambdalist, alpha = 1, nfolds=ncv, foldid=cvgroups)
plot(cvLASSOglm$lambda,cvLASSOglm$cvm,type="l",lwd=2,col="red",xlab="lambda",ylab="CV(10)",
     xlim=c(0,1),ylim = c(0,0.2))
whichlowestcvLASSO = order(cvLASSOglm$cvm)[1]; min(cvLASSOglm$cvm)
bestlambdaLASSO = lambdalist[whichlowestcvLASSO]; bestlambdaLASSO
abline(v=bestlambdaLASSO)

#ENET alpha=0.95 cross-validation
cvENET95glm = cv.glmnet(x, y, lambda=lambdalist, alpha = 0.95, nfolds=ncv, foldid=cvgroups)
plot(cvENET95glm$lambda,cvENET95glm$cvm,type="l",lwd=2,col="red",xlab="lambda",ylab="CV(10)",
     xlim=c(0,1),ylim = c(0,0.2))
whichlowestcvENET95 = order(cvENET95glm$cvm)[1]; min(cvENET95glm$cvm)
bestlambdaENET95 = lambdalist[whichlowestcvENET95]; bestlambdaENET95; abline(v=bestlambdaENET95)

#ENET alpha=0.5 cross-validation
cvENET50glm = cv.glmnet(x, y, lambda=lambdalist, alpha = 0.50, nfolds=ncv, foldid=cvgroups)
plot(cvENET50glm$lambda,cvENET50glm$cvm,type="l",lwd=2,col="red",xlab="lambda",ylab="CV(10)",
     xlim=c(0,1),ylim = c(0,0.2))
whichlowestcvENET50 = order(cvENET50glm$cvm)[1]; min(cvENET50glm$cvm)
bestlambdaENET50 = lambdalist[whichlowestcvENET50]; bestlambdaENET50; abline(v=bestlambdaENET50)

# A look at the lowest CVs among all methods.
min(cvRRglm$cvm)
min(cvLASSOglm$cvm)
min(cvENET95glm$cvm)
min(cvENET50glm$cvm)
#The diffs don't really matter. Ridge Regression was only a tiny bit better.

#fit selected model
Bestfit = glmnet(x, y, alpha = 0,lambda=lambdalist)
coef(Bestfit,s=bestlambdaRR) # 
plot(Bestfit,xvar="lambda"); abline(v=bestlambdaRR)
plot(Bestfit) #another way to look at it.


