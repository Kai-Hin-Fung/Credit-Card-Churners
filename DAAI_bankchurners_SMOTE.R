setwd("C:/Users/james/Desktop/DAAI files to submit")

BankChurners<-read.csv("all_numeric.csv")

#install.packages("smotefamily")
library(smotefamily)
library(caret)

BankChurners$Attrition_Flag = ifelse(BankChurners$Attrition_Flag=="Attrited Customer",1 ,0)

SMOTE <- SMOTE(X = BankChurners[,], target = BankChurners$Attrition_Flag, K = 10, dup_size = 3)

oversampled_data = SMOTE$data

oversampled_data <- oversampled_data[, -c(1,2,3,24)]

str(oversampled_data)

oversampled_data$Marital_Status <- factor( oversampled_data$Marital_Status, levels = c(1,2,3), labels = c("Single", "Married", "Divorced"))

set.seed(222) #144
split <- createDataPartition(oversampled_data$Attrition_Flag, p = 0.7, list = FALSE)
train.df <- oversampled_data[split,] 
test.df <- oversampled_data[-split,] 

prop.table(table(oversampled_data$Attrition_Flag))

#Normalizing the data
train.norm.df <- train.df
valid.norm.df <- test.df
oversampled_data.norm.df <- oversampled_data

norm.values <- preProcess(train.df[, 13:20], method=c("center", "scale")) 
train.norm.df[, 13:20] <- predict(norm.values, train.df[, 13:20])
valid.norm.df[, 13:20] <- predict(norm.values, test.df[, 13:20])
oversampled_data.norm.df[, 13:20] <- predict(norm.values, oversampled_data[, 13:20])
new.norm.df <- predict(norm.values, oversampled_data)

library(rpart)
library(rpart.plot)
library(caret)
library(e1071)

train.norm.df <- train.norm.df[, -c(2,5,8,9,13,15,16,20)]

logit.reg <- glm(Attrition_Flag ~ ., data = train.norm.df, family = "binomial") 
options(scipen = 999)
summary(logit.reg)

predicted <- predict(logit.reg, valid.norm.df[, -1], type = "response")
data.frame(actual = valid.norm.df$Attrition_Flag[1:10], predicted = predicted[1:10])

predicted <- ifelse(predicted >0.5, 1,0)

library(caret)
library(InformationValue)
library(ISLR)

confusionMatrix(predicted, valid.norm.df$Attrition_Flag)

sensitivity(as.factor(valid.norm.df$Attrition_Flag), predicted)
specificity(as.factor(valid.norm.df$Attrition_Flag), predicted)
optimal <- optimalCutoff(as.factor(valid.norm.df$Attrition_Flag), predicted)[1]
misClassError(as.factor(valid.norm.df$Attrition_Flag), predicted, threshold=optimal)

# first 10 actual and predicted records


default.ct <- rpart(Attrition_Flag ~ ., data = train.norm.df, method = "class")
prp(default.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10)


# set argument type = "class" in predict() to generate predicted class membership.
default.ct.point.pred.valid <- predict(default.ct,valid.norm.df,type = "class")
# generate confusion matrix for validation data
confusionMatrix(default.ct.point.pred.valid, valid.norm.df$Attrition_Flag)
            

#KNN 

upsampled_data = trainSMOTE$data
upsampled_data <- upsampled_data[, -c(1,2,3,24)]


set.seed(222) #144
split <- createDataPartition(upsampled_data$Attrition_Flag, p = 0.7, list = FALSE)
train_knn <- upsampled_data[split,] 
#train_knn <- train_knn[,2:21] #delete the right 2 columns
test_knn <- upsampled_data[-split,] 
#test_knn <- test[,2:21] #delete the right 2 columns
train_knn <- train_knn[, -c(2,5,7,8,9,13,15,16,20)]
train_knn$Attrition_Flag <- ifelse(train_knn$Attrition_Flag == 1, "Attrited_Customer", "Existing_CUstomer")

set.seed(2)
x = trainControl(method = "repeatedcv",
                 number = 10,
                 repeats = 3,
                 classProbs = TRUE,
                 summaryFunction = twoClassSummary)

knnModel <- train(Attrition_Flag~. , data = train_knn, method = "knn",
                  preProcess = c("center","scale"),
                  trControl = x,
                  metric = "ROC",
                  tuneLength = 10)
knnModel
plot(knnModel)

# Validation
test_knn_pred <- predict(knnModel,test_knn)
confusionMatrix(test_knn_pred, test_knn$Attrition_Flag)

test_knn_pred <- predict(knnModel,test_knn, type = "prob")
library(pROC)
r <- roc(test_knn$Attrition_Flag, test_knn_pred[,1]) 
plot.roc(r)
auc(r)

#neural networks
BankChurners_nn<- BankChurners

BankChurners_nn$Attrited_Customer <- BankChurners_nn$Attrition_Flag == 1
BankChurners_nn$Existing_Customer <- BankChurners_nn$Attrition_Flag == 0


str(BankChurners_nn)

BankChurners_nn <- BankChurners_nn[, -c(1,2,3)]

set.seed(2) #generate random data
train.index <- sample(c(1:dim(BankChurners_nn)[1]), dim(BankChurners_nn)[1]*0.7)  
train.df <- BankChurners_nn[train.index, ]
valid.df <- BankChurners_nn[-train.index, ]

train.norm.df <- train.df
valid.norm.df <- valid.df
BankChurners_nn.norm.df <- BankChurners_nn

# use preProcess() from the caret package to normalize Income and Lot_Size.
# "center": subtract mean from values;"scale": divide values by standard deviation.
norm.values <- preProcess(train.df[, 13:20], method=c("center", "scale")) 
train.norm.df[, 13:20] <- predict(norm.values, train.df[, 13:20])
valid.norm.df[, 13:20] <- predict(norm.values, valid.df[, 13:20])
BankChurners_nn.norm.df[, 13:20] <- predict(norm.values,BankChurners_nn[, 13:20])

library(neuralnet)

train.norm.df <- train.norm.df[, -c(1,2,3,4,5,6,7,8,9,10,11,12)]


nn <- neuralnet(Attrited_Customer + Existing_Customer ~ .
                , data = train.norm.df, linear.output = F, act.fct = "logistic", hidden = 3)


nn$weights

prediction(nn)

plot(nn, rep=1)

nn1 <- neuralnet(Attrited_Customer + Existing_Customer ~ . , data = train.norm.df, linear.output = F, rep = 100, act.fct = "logistic", hidden = 3)

nn1$weights

prediction(nn1)

plot(nn1, rep = "best")


