###############################################################################
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ Ingesting & Preprocessing Datasets ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
###############################################################################
rm(list=ls(all=TRUE))
#Current working directory
getwd()
#Set working directory to where your data files are stored
setwd("C:/Users/Desktop/Titanic Machine Learning") 
library(ggplot2)
library(ggrepel)

#Import the data and read the file
traindata <- read.csv(file="train.csv", header=TRUE,na.strings=c(""))
testdata <- read.csv(file="test.csv", header = TRUE, na.strings=c(""))
survived <- traindata$Survived
combined_data <- rbind(traindata[,-2],testdata)

#Determine the class and structure of train dataset
class(combined_data)
str(combined_data)
sapply(combined_data, function(x) sum(is.na(x)))

#Preprocessing the Dataset
combined_data$Pclass <- as.factor(combined_data$Pclass)
combined_data$Age <- as.integer(combined_data$Age)
combined_data$Name <- as.character(combined_data$Name)
str(combined_data)

#Feature Engineering
combined_data$Title <- gsub('(.*, )|(\\..*)', '', combined_data$Name)
combined_data$Title <- as.factor(combined_data$Title)
str(combined_data$Title)

combined_data$family_size <- as.integer(combined_data$SibSp+combined_data$Parch+1)
#Drop Cabin since more than 70% values are missing
#Drop some of the predictors that are redundant
combined_data <- subset(combined_data,select = -c(Name,SibSp,Parch,Ticket,Cabin))
combined_data$Fare <- as.integer(combined_data$Fare)
combined_data[which(combined_data$Age==0),] #None title as Mr, Mrs, or above 18
combined_data$Age[which(combined_data$Age==0)] <- 1

levels(combined_data$Title)
sort(summary(combined_data$Title)[1:18],decreasing = T)
var <- c("Mr","Miss","Mrs","Master")
length(var);rev(var)

for(i in rev(var)){
  combined_data$Title <- relevel(combined_data$Title, i)
}
#Grouping the remaining speakers as 'others'
levels(combined_data$Title) <- c("Mr","Miss","Mrs","Master", rep("VIP",14))
qplot(combined_data$Title)


#Imputation of missing values
#Handle NAs in the dataset
sapply(combined_data,function(x) sum(is.na(x)))
#Use the concept of Random Forest to calculate missing values for quantitative
#and qualitative predictors. It's a non-parametric approach without any assumptions
#install.packages("missForest")
library(missForest)
combined_data.imp <- missForest(combined_data)
sum(is.na(combined_data.imp$ximp))
#Shows the error rate during computation of missing values
#Quite an effective way to check quality of computational process
combined_data.imp$OOBerror
combined_data <- combined_data.imp$ximp
str(combined_data)

traindata <- combined_data[1:891,]
traindata$Survived <- survived
testdata <- combined_data[892:1309,]

###############################################################################
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ Visualizing the Datasets ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
###############################################################################
#Plotting relation between Age & Survived on traindata
ggplot(traindata, aes(x= Age, fill=factor(Survived))) + 
  geom_bar(width=0.5) +
  xlab("Age") + 
  ylab("Total Count") + 
  labs(fill="Survived")

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

###############################################################################
#~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ Logistic Regression ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
###############################################################################
#Splitting the dataset into 80:20 Train and Test.
size_train <- nrow(traindata)
sample_index <- sample.int(size_train, size = floor(0.2*nrow(traindata)))
testData <- traindata[sample_index,]
trainData <- traindata[-sample_index,]

library(e1071)
#To verify that the model is reasonable, we calculate the data fitting for 
#both training set and cross validation set.
logmodel <- glm(factor(Survived)~., data = testData, family = binomial(link = logit), control = list(maxit=15,epsilon=1e-4))
cutoffs <- seq(0.1,0.9,0.1)
accuracy <- NULL

#Function to Set accuracy for each cutoffs
for (i in seq(along = cutoffs)){
  prediction <- ifelse(logmodel$fitted.values >= cutoffs[i], 1, 0) #Predicting for cut-off
  accuracy <- c(accuracy,length(which(factor(testData$Survived) ==prediction))/length(prediction)*100)
}

plot(cutoffs, accuracy, pch =19,type='b',col= "steelblue",
     main ="Logistic Regression", xlab="Cutoff Level", ylab = "Accuracy %")

#From the plot above the best Accuracy % is at cutoff 0.50

full_mod <- glm(factor(Survived)~., data = trainData, family = binomial(link = logit), control = list(maxit=15,epsilon=1e-4))
summary(full_mod)
library(MASS)
backward <- stepAIC(full_mod,direction="backward",trace=FALSE)
#Building the model with lowest deviance and lowest AIC
mylogit <- glm(formula = factor(Survived) ~ Pclass + Sex + Age + Fare + 
                 Title + family_size, family = binomial(link = logit), data = trainData, control = list(maxit = 15, epsilon = 1e-04))

summary(mylogit)

predSurv <- predict.glm(mylogit,testdata)
predSurv <-ifelse(predSurv > 0.5,1,0)

final_result <- data.frame(testdata$PassengerId,predSurv)
colnames(final_result) <- c("PassengerId","Survived")
write.csv(final_result, file="logistic.csv", row.names=FALSE)


###############################################################
#----------------------------- SVM ---------------------------#
###############################################################
str(combined_data)
SVMmodel<-svm(factor(Survived) ~ Pclass + Sex + Age + Fare + 
                Title + family_size, data = traindata, kernel="radial", cost = 100, gamma = 1)
print(SVMmodel)
summary(SVMmodel)
compareTable <- table(predict(SVMmodel),traindata$Survived)  
mean(traindata$Survived != predict(SVMmodel)) #10.7% misclassification error

SVMmodel<-svm(factor(Survived) ~ Pclass + Sex + Age + Fare + 
                Title + family_size, data = traindata, kernel="linear", cost = 100, gamma = 1)
print(SVMmodel)
summary(SVMmodel)
compareTable <- table(predict(SVMmodel),traindata$Survived)  
mean(traindata$Survived != predict(SVMmodel)) #16.8% misclassification error
# Tuning
# Prepare training and test data
set.seed(100) # for reproducing results
sum(is.na(traindata))
rowIndices <- 1 : nrow(traindata) # prepare row indices i.e. 891
sampleSize <- 0.7 * length(rowIndices) # training sample size i.e 623 = 0.7*891
trainingRows <- sample (rowIndices, sampleSize) # random sampling of 623 rows
trainingData <- traindata[trainingRows, ] # training data i.e. random sample 623 values
testData <- traindata[-trainingRows, ] # test data i.e. rando sample 267 values

tuned <- tune.svm(factor(Survived) ~ Pclass + Sex + Age + Fare + 
                    Title + family_size, data = traindata, gamma = 10^(-6:-1), cost = 10^(1:3)) # tune

summary (tuned) # to select best gamma and cost

svmfit <- svm (factor(Survived) ~., data = traindata, kernel = "radial", cost = 100, gamma=0.01) # radial svm, scaling turned OFF
print(svmfit)
compareTable <- table (testData$Survived, predict(svmfit, testData))  # comparison table
mean(testData$Survived != predict(svmfit, testData)) # 14.55% misclassification train error
1-0.1455 #85.45% Accuracy
#Testing on unseen data = testdata
prediction1 <- predict(svmfit, testdata)
final_result <- data.frame(testdata$PassengerId,prediction1)
colnames(final_result) <- c("PassengerId","Survived")
write.csv(final_result, file="svm.csv", row.names=FALSE)

###############################################################
#--------------------------- Decision Tree -------------------#
###############################################################
#Predict Survival (classify)
dim(traindata)
train_data <- traindata[,-c(1)]
str(train_data)

dim(testdata)
PassengerId <- testdata[,1]
test_data <- testdata[,-c(1)]
str(test_data)

train_data$Survived <- as.factor(train_data$Survived)

#C5.0 good classification technique but does not 
#provide good plots. For that rpart is good
library(C50)
Model_C50 <-C5.0(train_data[,-8],train_data[,8]) #Without Survived, Survived
#Here 8 is column for 'Survived'
Model_C50
summary(Model_C50)

#Predicting on Train
P1_train=predict(Model_C50,train_data);P1_train
table(train_data[,8],Predicted=P1_train)    #Here 8 is column for 'Survived'
(516+251)/(516+33+91+251)
#86.08

#Predicting on Test
P1_test = predict(Model_C50,test_data);P1_test
final_result <- data.frame(PassengerId,P1_test)
colnames(final_result) <- c("PassengerId","Survived")
write.csv(final_result, file="MC50.csv", row.names=FALSE)

#Rpart
library(rpart)
library(rpart.plot)

Model_rpart= rpart(factor(Survived)~.,data=train_data, method="class")
Model_rpart
summary(Model_rpart)
#plot(Model_rpart)
#rpart.plot(Model_rpart,type=3,extra=101,fallen.leaves = FALSE)

#Predicting on Train
P1_train_rpart=predict(Model_rpart,train_data,type="class")
table(train_data[,8],predicted=P1_train_rpart)  #Here 8 is column for 'Survived'
(477+251)/(477+72+91+251) # 81.70

#Predicting on Test
P1_test_rpart=predict(Model_rpart,test_data,type="class")
final_result <- data.frame(PassengerId, P1_test_rpart)
names(final_result) <- c("PassengerId","Survived")
write.csv(final_result,file = "Rpart.csv", row.names = FALSE)



###############################################################
#--------------------- RANDOM FOREST -------------------------#
###############################################################
#Grid Search for Tuning Hyperparameters using CARET
library(caret)
library(randomForest)
y <- as.factor(survived)
bestmtry <- tuneRF(traindata,y, stepFactor=1.5, improve=1e-5, ntreeTry = 500)
print(bestmtry)
rf1 <- randomForest(factor(survived) ~ ., data=traindata[-1], keep.forest=TRUE, ntree=500, mtry=2)
print(rf1) #OOB estimate of  error rate: 16.72%
varImpPlot(rf1)
#confusion Matrix
table(survived, predict(rf1, traindata[,-1], type="response", norm.votes=TRUE))
RFpred <- predict(rf1, testdata[,-1], type="response", norm.votes=TRUE)
final_result <- data.frame(testdata$PassengerId, RFpred)
names(final_result) <- c("PassengerId","Survived")
write.csv(final_result,file = "RF.csv", row.names = FALSE)

###############################################################
#------------------------- XG BOOOST -------------------------#
###############################################################
str(traindata)
str(testdata)
library(xgboost)
library(mlr)
class(traindata$Pclass)
traindata$Pclass <- as.numeric(as.character(traindata$Pclass))
traindata$Title <- createDummyFeatures(traindata$Title)
traindata$Embarked <- createDummyFeatures(traindata$Embarked)
traindata$Sex <- createDummyFeatures(traindata$Sex)

testdata$Pclass <- as.numeric(as.character(testdata$Pclass))
testdata$Title <- createDummyFeatures(testdata$Title)
testdata$Embarked <- createDummyFeatures(testdata$Embarked)
testdata$Sex <- createDummyFeatures(testdata$Sex)


bst <- xgboost(data=as.matrix(traindata),
               label = survived, max_depth=5, eta=0.5, 
               nthread=3, nrounds = 4, eval_metric = "error", 
               objective="binary:logistic", verbose = 1)
pred <- predict(bst,as.matrix(testdata))
print(length(pred))
print(head(pred))
prediction <- as.numeric(pred>0.5)
print(head(prediction))
final_result <- data.frame(testdata$PassengerId, prediction)
names(final_result) <- c("PassengerId","Survived")
write.csv(final_result,file = "XGB.csv", row.names = FALSE)
