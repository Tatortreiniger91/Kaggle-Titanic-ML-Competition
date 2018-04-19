# load necessary packages
# install.packages("caTools")
# install.packages("pROC")
# install.packages("gbm")
library(car)
library(dplyr)
library(caret)
library(caTools)
library(pROC)
library(rpart)
library(gbm)


# clean environment
rm(list=ls())

# take the time
ptm_Skript_beginn <- proc.time()

# set path and working directory
mainDir <- "path/Kaggle/Titanic"
setwd(file.path(mainDir))

# load train, test and testsubmission datasets
train <- read.csv("train.csv", header = T)
test <- read.csv("test.csv", header = T)

# load testsubmission
testsubmission <- read.csv("sample_submission.csv", header = T)

#safe the PassengerId for later submissions
PassengerId <- test$PassengerId

#create an Survived variable in the test set
test$Survived <- numeric(nrow(test))

# combine the train and test set for pre-processing (drop PassengerId)
whole <- rbind(train[,-1], test[,-1])
summary(whole)

# set NAs in Embarked, impute Age and Fare by mean and recode Sex
whole[,"Embarked"][which(is.na(whole[,"Embarked"]))] <- NA
whole[,"Age"][which(is.na(whole[,"Age"]))] <- mean(whole$Age, na.rm = TRUE)
whole[,"Fare"][which(is.na(whole[,"Fare"]))] <- mean(whole$Fare, na.rm = TRUE)
whole$Sex <- recode(whole$Sex, "female"=1, "male"=0)

# extract the passengers title out of the Name
whole$title <- gsub('.*, (.*)', '\\1',whole$Name)
whole$title <- gsub('. (.*)','\\2',whole$title)

# combine titles in groups as dummy variables
whole$Miss <- ifelse(whole$title == "Miss"| whole$title == "Ms" | whole$title == "Mlle", 1, 0)
whole$Mrs <- ifelse(whole$title == "Mrs"| whole$title == "Mme" | whole$title == "Lady"| whole$title == "Dona", 1, 0)
whole$Mr <- ifelse(whole$title == "Mr"| whole$title == "Don" | whole$title == "Sir" | whole$title == "Jonkheer", 1, 0)
whole$high <- ifelse(whole$title == "Dr"| whole$title == "Master", 1, 0)
whole$military <- ifelse(whole$title == "Col"| whole$title == "Major", 1, 0)
whole$other <- ifelse(whole$title == "Capt"| whole$title == "Rev" | whole$title == "th", 1, 0)

# create unique levels
feature.names <- names(whole)

for (f in feature.names) {
  if (class(whole[[f]])=="factor") {
    levels <- unique(c(whole[[f]]))
    whole[[f]] <- factor(whole[[f]],
                         labels=make.names(levels))
  }
}

# drop unused variables (cabin class could be important for the prediction but have to be cleaned before)
whole$Ticket <- NULL
whole$Cabin <- NULL
whole$title <- NULL
whole$Name <- NULL

# split in train and test dataset and check
train2 <- whole[1:891,]
test2 <- whole[892:1309,]
summary(train2)
summary(test2)

modelcompare <- numeric(4)

##############Models##############
set.seed(123)
model1 <- glm(Survived~., data=train2, family = "binomial")
p1 <- predict(model1, test2, type="response")
p_class1 <- ifelse(p1 > 0.5, 1, 0)
table(p_class1, testsubmission$Survived)
accuracy <- mean(p_class1 == testsubmission$Survived)
modelcompare[1] <- accuracy
Survived <- as.numeric(p_class1)
submission1 <- data.frame(PassengerId, Survived)
write.csv(submission1, "submission1.csv", row.names = F)
# submission1 kaggle score: 0.77990

set.seed(123)
model2 <- rpart(
  Survived~., data=train2,
  method = "class"
)
plotcp(model2)
model2$cptable
model_pruned <- prune(model2, cp = 0.01000000)
Survived <- predict(model_pruned, test2, type="class")
table(Survived, testsubmission$Survived)
accuracy <- mean(Survived == testsubmission$Survived)
modelcompare[2] <- accuracy
submission2 <- data.frame(PassengerId, Survived)
write.csv(submission2, "submission2.csv", row.names = F)
# submission3 kaggle score: 0.77990



# for glm we need factor variables
train2$Survived <- as.factor(train2$Survived)
test2$Survived <- as.factor(test2$Survived)
levels <- unique(c(train2[["Survived"]]))
train2[["Survived"]] <- factor(train2[["Survived"]],
                     labels=make.names(levels))
levels <- unique(c(test2[["Survived"]]))
test2[["Survived"]] <- factor(test2[["Survived"]],
                               labels=make.names(levels))

set.seed(123)
model3 <- train(Survived~.,
                  data=train2,
                  method = "glmnet",
                  trControl = trainControl(
                    method = "cv",
                    number = 10,
                    summaryFunction = twoClassSummary,
                    classProbs = TRUE, # IMPORTANT!
                    verboseIter = TRUE),
                  preProcess = c("center", "scale", "pca")
)
Survived <- ifelse(predict(model3, test2, type="raw")=="X2",1,0)
table(Survived, testsubmission$Survived)
accuracy <- mean(Survived == testsubmission$Survived)
modelcompare[3] <- accuracy
submission3 <- data.frame(PassengerId, Survived)
write.csv(submission3, "submission3.csv", row.names = F)
# submission4 kaggle score: 0.79425



# for gbm we need numeric data so let's recode them
train2$Survived <- ifelse(train2$Survived == "X2", 1, 0)
test2$Survived <- ifelse(test2$Survived == "X2", 1, 0)
set.seed(123)
model4 <- gbm(formula = Survived ~ ., 
             distribution = "bernoulli",
             data = train2,
             n.trees = 10000,
             cv.folds = 10
)

#get the optimal number of trees
ntree_opt_cv <- gbm.perf(model4, method = "cv")
ntree_opt_oob <- gbm.perf(model4, method = "OOB")

set.seed(123)
model4_new <- gbm(formula = Survived ~ ., 
             distribution = "bernoulli",
             data = train2,
             n.trees = min(ntree_opt_cv, ntree_opt_oob),
             cv.folds = 10
)
print(model4_new)
summary(model4_new)
p2 <- predict(model4_new, newdata = test2, type = "response", n.trees = min(ntree_opt_cv, ntree_opt_oob))
Survived <- ifelse(p2 > 0.5, 1, 0)
table(Survived, testsubmission$Survived)
accuracy <- mean(Survived == testsubmission$Survived)
modelcompare[4] <- accuracy
submission4 <- data.frame(PassengerId, Survived)
write.csv(submission4, "submission4.csv", row.names = F)
# submission5 kaggle score: 0.77511

# check the skript running time
modelcompare
ptm_Skript_end <- proc.time()-ptm_Skript_beginn
ptm_Skript_end[[3]]
