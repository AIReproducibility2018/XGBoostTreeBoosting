import xgboost as xgb

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
import time


def calculateAccuracy(correct, pred):
    sum = 0
    correct0 = 0
    correct1 = 0
    pred0 = 0
    pred1 = 0
    j = 200
    for i in range(len(correct)):
        #if j > 0:
        #    print("Correct: " + str(correct[i]) + " : Pred: " + str(pred[i]))
        #    j-=1
        if correct[i] == 1:
            correct1 += 1
        if correct[i] == 0:
            correct0 += 1
        if pred[i] == 1:
            pred1 += 1
        if pred[i] == 0:
            pred0 += 1
        if correct[i] == pred[i]:
            sum += 1
    print("Correct0: " + str(correct0) + " Correct1: " + str(correct1) + " Pred0: " + str(pred0) + " Pred1: " + str(pred1) + " Sum: " + str(sum) + " len(correct): " + str(len(correct)) + " len(pred): " + str(len(pred)))
    return float(sum) / len(correct)


t0 = time.time()
df = pd.read_csv('D:\Development\MasterProject\DataSets\Higgs\HIGGS.csv\HIGGS.csv')
#df = pd.read_csv('D:\Development\MasterProject\DataSets\Higgs\HIGGS.csv\HiggsTenth.csv')
t1 = time.time()
total1 = t1-t0
print("Read done: " + str(total1))

results = [0.0]
roundNr = 1

while roundNr < 7:

    df['split'] = np.random.randn(df.shape[0], 1)

    msk = np.random.rand(len(df)) <= 0.90909090909

    train = df[msk]
    test = df[~msk]


    for f in train.columns:
        if train[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values))
            train[f] = lbl.transform(list(train[f].values))

    t2 = time.time()
    total2 = t2-t1
    #print("Train fixed: " + str(total2))

    for f in test.columns:
        if test[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(test[f].values))
            test[f] = lbl.transform(list(test[f].values))

    t3 = time.time()
    total3 = t3-t2
    #print("Test fixed: " + str(total3))

    train.fillna((-999), inplace=True)
    test.fillna((-999), inplace=True)

    trainTemp = np.array(train)
    testTemp = np.array(test)
    trainNP = trainTemp.astype(float)
    testNP = testTemp.astype(float)

    trainX = trainNP[:, 1:]
    trainY = trainNP[:, 0]
    testX = testNP[:, 1:]
    testY = testNP[:, 0]

    #print("Shape trainX: " + str(trainX.shape) + " trainY: " + str(trainY.shape))
    #print("Shape testX: " + str(testX.shape) + " testY: " + str(testY.shape))

    #data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
    #label = np.random.randint(2, size=5)  # binary target
    #dtrain = xgb.DMatrix(data, label=label)
    #data2 = np.random.rand(5, 10)  # 5 entities, each contains 10 features
    #label2 = np.random.randint(2, size=5)  # binary target
    #dtest = xgb.DMatrix(data, label=label)

    dtrain = xgb.DMatrix(trainX, label=trainY)
    dtest = xgb.DMatrix(testX, label=testY)
    # dtrain = xgb.DMatrix(train)#xgb.DMatrix("D:\Development\MasterProject\DataSets\Higgs\HIGGS.csv\hformattedHiggs")
    # dtest = xgb.DMatrix(test)
    param = {
        'n_estimators': 500,
        #'colsample_bytree': 0.8,
        'silent': 1,
        'max_depth': 8,
        #'min_child_weight': 1,
        'learning_rate': 0.1,
        #'subsample': 0.8,
        'eta': 0.1,
        #'objective': 'multi:softmax',
        #'num_class': 2,
        'objective': 'binary:logistic',
        'tree_method': 'exact',
        'eval_metric': 'auc',
        #'tree_method': 'auto'
    }
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = 200


    t4 = time.time()
    total4 = t4-t3
    #print("Starting Training: " + str(total4))
    plst = param.items()
    bst = xgb.train(param, dtrain, num_round)

    t5 = time.time()
    total5 = t5-t4
    print("End Training: " + str(total5))

    bst.dump_model('dump.raw.txt', 'featmap.txt')

    t6 = time.time()
    total6 = t6-t5
    #print("Dump model: " + str(total6))

    preds = bst.predict(dtest)
    #preds = bst.predict(dtrain)

    #print("preds: " + str(preds.item(0)))

    best_preds = []

    greater0 = False
    greater1 = False
    for p in preds:
        if p < 0.5:
            best_preds.append(0)
        else:
            best_preds.append(1)

    #print("greater0 and greater1: " + str(greater0) + " : " + str(greater1))

    #best_preds = np.asarray([np.argmax(line) for line in preds])

    t7 = time.time()
    total7 = t7-t6
    #print("Prediction: " + str(total7))

    correctPred = []
    for e in testY:
        correctPred.append(float(e))

    formattedPreds = best_preds

    #xgb.plot_importance(bst)

    auc = roc_auc_score(correctPred, preds, average='micro')
    results.append(auc)

    print("Result Round " + str(roundNr) + ": " + str(auc))
    #print("Result: " + str(calculateAccuracy(correctPred, formattedPreds)))

    roundNr += 1

    joblib.dump(bst, 'bst_model.pkl', compress=True)
