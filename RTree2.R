#install.packages("gbm")
#install.packages("cvAUC")
library(gbm)
library(cvAUC)
# Load 2-class HIGGS dataset
train <- read.csv(file="D:/Development/MasterProject/DataSets/Higgs/HIGGS.csv/trainSmall.csv", header=FALSE)
test <- read.csv("D:/Development/MasterProject/DataSets/Higgs/HIGGS.csv/trainSmall.csv")
set.seed(1)
dataTrain = as.matrix(train)
model <- gbm(formula = class ~ . - train$V28, 
             distribution = "bernoulli",
             data = train,
             n.trees = 500,
             interaction.depth = 5,
             shrinkage = 0.3,
             n.minobsinnode = 0.1,
             bag.fraction = 0.5,
             train.fraction = 1.0,
             n.cores = NULL)  #will use all cores by default
print(model)
# Generate predictions on test dataset
preds <- predict(model, newdata = test, n.trees = 70)
labels <- test[,"response"]

# Compute AUC on the test set
cvAUC::AUC(predictions = preds, labels = labels)