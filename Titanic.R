###############################################################################
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ Ingesting & Preprocessing Datasets ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
###############################################################################
rm(list=ls(all=TRUE))
gc()

#Current working directory
getwd()

# Step 1: Set working directory to where your data files are stored
setwd("C:\\Users\\Nitin.Chauhan\\Documents\\GitHub\\Kaggle-Titanic-MachineLearning")
library(ggplot2)
library(ggrepel)

# Step 2: Import the data and read the file
traindata <- read.csv(file="train.csv", header=TRUE,na.strings=c(""))
testdata <- read.csv(file="test.csv", header = TRUE, na.strings=c(""))
# Save the Survived labels from train data in variable survived
survived <- traindata$Survived

# Step 3: Determine the class and structure of train dataset
class(traindata)
sapply(traindata, function(x) sum(is.na(x)))
# Preprocessing the Train Dataset
traindata$Pclass <- as.factor(traindata$Pclass)
traindata$Age <- as.integer(traindata$Age)
traindata$Name <- as.character(traindata$Name)
str(traindata)

# Step 4: Determine the class and structure of test dataset
class(testdata)
sapply(testdata, function(x) sum(is.na(x)))
# Preprocessing the Test Dataset
testdata$Pclass <- as.factor(testdata$Pclass)
testdata$Age <- as.integer(testdata$Age)
testdata$Name <- as.character(testdata$Name)
str(testdata)

# Step 5: Feature Engineering in Train Data
# Create a new feature Title
traindata$Title <- gsub('(.*, )|(\\..*)', '', traindata$Name)
traindata$Title <- as.factor(traindata$Title)
str(traindata$Title)
# Create a new feature family_size
traindata$family_size <- as.integer(traindata$SibSp + traindata$Parch + 1)

"Drop Cabin since more than 70% values are missing
Drop some of the predictors that are redundant"
traindata <- subset(traindata,select = -c(Name,SibSp,Parch,Ticket,Cabin))

# Identify if any Age is identified as 0
traindata[which(traindata$Age==0),] # Where Age is less than a year either new born or wrongly interpreted
traindata$Age[which(traindata$Age==0)] <- 1 # Since Title are either Master or Miss

levels(traindata$Title)
sort(summary(traindata$Title)[1:17],decreasing = T)
# Create a new variable var holding few titles
var <- c("Mr", "Mrs", "Master", "Miss", "Dr")
# Since there are some titles that are 1 in count causing the system to see some unseen title in test data
length(traindata$Title[!traindata$Title %in% var]) # 20
# Rename the 20 other titles as VIP
traindata$Title <- as.character(traindata$Title)
traindata$Title[!traindata$Title %in% var] <- 'VIP'
traindata$Title <- as.factor(traindata$Title)

# Step 6: Feature Engineering in Test Data
# Create a new feature Title
testdata$Title <- gsub('(.*, )|(\\..*)', '', testdata$Name)
testdata$Title <- as.factor(testdata$Title)
str(testdata$Title)
# Create a new feature family_size
testdata$family_size <- as.integer(testdata$SibSp + testdata$Parch + 1)

"Drop Cabin since more than 70% values are missing
Drop some of the predictors that are redundant"
testdata <- subset(testdata,select = -c(Name,SibSp,Parch,Ticket,Cabin))

# Identify if any Age is identified as 0
testdata[which(testdata$Age==0),] # Where Age is less than a year either new born or wrongly interpreted
testdata$Age[which(testdata$Age==0)] <- 1 # Since Title are either Master or Miss

levels(testdata$Title)
sort(summary(testdata$Title)[1:9],decreasing = T)
# Since there are some titles that are 1 in count causing the system to see some unseen title in test data
length(testdata$Title[!testdata$Title %in% var]) # 6 
# Rename the 6 other titles as VIP
testdata$Title <- as.character(testdata$Title)
testdata$Title[!testdata$Title %in% var] <- 'VIP'
testdata$Title <- as.factor(testdata$Title)

"Use the concept of Random Forest to calculate missing values for quantitative
and qualitative predictors. It's a non-parametric approach without any assumptions"
#install.packages("missForest")
library(missForest)

# Step 7: Impute Missing Values in Train Data
sapply(traindata,function(x) sum(is.na(x)))
traindata.imp <- missForest(traindata)
sum(is.na(traindata.imp$ximp))
"Shows the error rate during computation of missing values
Quite an effective way to check quality of computational process"
traindata.imp$OOBerror
traindata <- traindata.imp$ximp

# Step 8: Impute Missing Values in Test Data
sapply(testdata,function(x) sum(is.na(x)))
testdata.imp <- missForest(testdata)
sum(is.na(testdata.imp$ximp))
"Shows the error rate during computation of missing values
Quite an effective way to check quality of computational process"
testdata.imp$OOBerror
testdata <- testdata.imp$ximp


# Step 9: Identify number of observation for each class
count_survived <- qplot(factor(traindata$Survived), fill = factor(traindata$Survived))
count_survived + geom_label_repel(stat='count', aes(label=..count..), nudge_x = 0.0, nudge_y = -50) + labs(title="No. of observation for each class",
                                                                                                           x ="Class", y = "Count")
"Since the labels are imbalanced in count, We will use Precision-Recall curves 
as it summarize the trade-off between the true positive rate and 
the positive predictive value for a predictive model using different probability thresholds."

###############################################################################
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ Visualizing the Datasets ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
###############################################################################
#Plotting relation between Age & Survived on traindata
ggplot(traindata, aes(x= Age, fill=factor(Survived))) + 
  geom_bar(width=0.5) + labs(title="Relation between Age & Survived", x ="Age", y = "Total Count", fill="Survived")
  

#Plotting relation between Sex & Survived on traindata
ggplot(traindata, aes(x= Sex, fill=factor(Survived))) + 
  geom_bar(width=0.5) + geom_label_repel(stat='count', aes(label=..count..), nudge_x = 0, nudge_y = 0.2) +
  xlab("Sex") + 
  ylab("Total Count") + 
  labs(fill="Survived")

#Plotting relation between Pclass & Survived on traindata
ggplot(traindata, aes(x= Pclass, fill=factor(Survived))) +  
  geom_bar(width=0.5) + geom_label_repel(stat='count', aes(label=..count..),nudge_x = 0, nudge_y = 0.2) +
  xlab("Pclass") + 
  ylab("Total Count") + 
  labs(fill="Survived")

#Plotting relation between Embarked & Survived on traindata
ggplot(traindata, aes(x= Embarked, fill=factor(Survived))) + 
  geom_bar(width=0.5) + geom_label_repel(stat='count', aes(label=..count..), nudge_x = 0, nudge_y = 0.2) +
  xlab("Embarked") + 
  ylab("Total Count") + 
  labs(fill="Survived")

#Plotting relation between Title and Pclass on traindata
ggplot(traindata, aes(x= Title, fill=factor(Pclass))) + 
  geom_bar(width=0.5) + geom_label_repel(stat='count', aes(label=..count..),nudge_x = 0, nudge_y = 0.2) +
  xlab("Title") + 
  ylab("Total Count") + 
  labs(fill="Pclass")


#Splitting the dataset into 70:30 Train and Valid
size_train <- nrow(traindata)
sample_index <- sample.int(size_train, size = floor(0.3*nrow(traindata)))
valid <- traindata[sample_index,]
train <- traindata[-sample_index,]


###############################################################################
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ Logistic Regression ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
###############################################################################
library(e1071)
#To verify that the model is reasonable, we calculate the data fitting for 
#both training set and cross validation set.
logmodel <- glm(factor(Survived)~., data = valid, family = binomial(link = logit), control = list(maxit=15,epsilon=1e-4))
cutoffs <- seq(0.1,0.9,0.1)
accuracy <- NULL

#Function to Set accuracy for each cutoffs
for (i in seq(along = cutoffs)){
  prediction <- ifelse(logmodel$fitted.values >= cutoffs[i], 1, 0) #Predicting for cut-off
  accuracy <- c(accuracy,length(which(factor(valid$Survived) ==prediction))/length(prediction)*100)
}

plot(cutoffs, accuracy, pch =19,type='b',col= "steelblue",
     main ="Logistic Regression", xlab="Cutoff Level", ylab = "Accuracy %")

#From the plot above the best Accuracy % is at cutoff 0.6

full_mod <- glm(factor(Survived)~., data = train, family = binomial(link = logit), control = list(maxit=15,epsilon=1e-4))
summary(full_mod)
library(MASS)
backward <- stepAIC(full_mod,direction="backward",trace=FALSE)
#Building the model with lowest deviance and lowest AIC
mylogit <- glm(formula = factor(Survived) ~ Pclass + Sex + Age + Fare + 
                 Title + family_size, family = binomial(link = logit), data = train, 
               control = list(maxit = 15, epsilon = 1e-04))

summary(mylogit)

# Test the model on valid
pred <- predict.glm(mylogit,valid)
pred <- ifelse(pred > 0.5,1,0)
nrow(valid)
length(pred)
library(caret)
xtab <- table(pred, valid$Survived)
print(confusionMatrix(xtab[2:1,2:1]))

# Prediction on TestData
predSurv <- predict.glm(mylogit,testdata)
predSurv <- ifelse(predSurv > 0.5,1,0)
length(predSurv)
final_result <- data.frame(testdata$PassengerId,predSurv)
colnames(final_result) <- c("PassengerId","Survived")
write.csv(final_result, file="logistic.csv", row.names=FALSE)

###############################################################
#----------------------------- SVM ---------------------------#
###############################################################
# Step 1: Compare Kernels based on Validation set
# Step 2: Radial Kernel
SVMmodel<-svm(factor(Survived) ~ ., data = valid, kernel="radial", cost = 100, gamma = 1)
print(SVMmodel)
summary(SVMmodel)
compareTable <- table(predict(SVMmodel),valid$Survived)  
mean(valid$Survived != predict(SVMmodel)) #1.12% misclassification error

# Step 3: Linear Kernel
SVMmodel<-svm(factor(Survived) ~ ., data = valid, kernel="linear", cost = 100, gamma = 1)
print(SVMmodel)
summary(SVMmodel)
compareTable <- table(predict(SVMmodel),valid$Survived)
mean(valid$Survived != predict(SVMmodel)) #16.47% misclassification error

"The Radial kernel seems to result in low misclassification error thus we will use it for further modeling"
# Step 4: Tune the model using tune.svm on validation set
tuned <- tune.svm(factor(Survived) ~ ., data = valid, gamma = 10^(-6:-1), cost = 10^(1:3)) # tune
summary (tuned) # to select best gamma and cost
model1 <- svm (factor(Survived) ~., data = valid, kernel = "radial", cost = 100, gamma=0.01) # radial svm, scaling turned OFF
print(model1)
compareTable <- table (valid$Survived, predict(model1, valid))  # comparison table
mean(valid$Survived != predict(model1, valid)) # 13.48% misclassification train error

# Step 5: Tune the model using k fold cross validation
train_control <- trainControl(method="cv", number=10)
# Fit SVM Radial Model
model <- train(factor(Survived) ~ ., data=valid, trControl=train_control, method="svmRadial")
# Summarise Results
print(model)
# C=0.50  Accuracy=0.8390313  K=0.6360905

# Step 6: Create a prediction model
svmfit <- train(factor(Survived) ~ ., data=train, trControl=train_control, method="svmRadial")
# Summarise Results
print(svmfit)

# Step 7: Testing on unseen data 'testdata'
prediction1 <- predict(svmfit, testdata)
length(prediction1)
final_result <- data.frame(testdata$PassengerId,prediction1)
colnames(final_result) <- c("PassengerId","Survived")
write.csv(final_result, file="svm.csv", row.names=FALSE)


###############################################################
#--------------------- RANDOM FOREST -------------------------#
###############################################################

# Step 1: Compare Different Tuning algorithms based on Validation set
# Step 2: Grid Search for Tuning Hyperparameters using CARET
library(randomForest)
y <- as.factor(valid$Survived)
bestmtry <- tuneRF(valid[,-2],y, stepFactor=1.5, improve=1e-5, ntreeTry = 500)
print(bestmtry)
# Step 3: Create a model using hyperparameters obtained from Grid Search
rf1 <- randomForest(factor(train$Survived) ~ ., data=train[-2], keep.forest=TRUE, ntree=500, mtry=2)
print(rf1) # OOB estimate of  error rate: 16.72%
varImpPlot(rf1)
# confusion Matrix
table(train$Survived, predict(rf1, train[,-2], type="response", norm.votes=TRUE))

# Step 4: Create a model using k-fold cross validation
train_control <- trainControl(method="cv", number=10)
# Fit SVM Radial Model
rf2 <- train(factor(Survived) ~ ., data=train, trControl=train_control, method="rf")
# Summarise Results
print(rf2)

# Compare the results of model rf1 & rf2 and which ever results in best result, use that model to predict on unseen data
# Step 5: Testing on unseen data 'testdata'
RFpred <- predict(rf2, testdata)
final_result <- data.frame(testdata$PassengerId, RFpred)
names(final_result) <- c("PassengerId","Survived")
write.csv(final_result,file = "RF.csv", row.names = FALSE)
