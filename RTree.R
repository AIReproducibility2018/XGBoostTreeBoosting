
train <- read.csv(file="D:/Development/MasterProject/DataSets/Higgs/HIGGS.csv/HiggsSmall.csv", header=FALSE, sep=",")
head(train)
randomSeed = 1337
set.seed(randomSeed)

library(gbm)

start.time <- Sys.time()

gbmModel = gbm(formula = formula(train),
               distribution = "gaussian",
               n.trees = 500,
               interaction.depth = 1,
               n.minobsinnode = 1,
               shrinkage = .01,
               bag.fraction = 0.5,
               train.fraction = 1.0,
               cv.folds=0,
               data = train,
               var.monotone = NULL,
               keep.data = TRUE,
               verbose = "CV",
               class.stratify.cv=NULL,
               n.cores = NULL,
               )



end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

