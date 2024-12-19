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

def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

t0 = time.time()
trainDf = pd.read_csv('C:\\Users\Axim-\Downloads\Yahoo\Webscope_C14\Webscope_C14\Learning to Rank Challenge\ltrc_yahoo.tar\ltrc_yahoo\set1.trainParsed.txt')
testDf = pd.read_csv('C:\\Users\Axim-\Downloads\Yahoo\Webscope_C14\Webscope_C14\Learning to Rank Challenge\ltrc_yahoo.tar\ltrc_yahoo\set1.testParsed.txt')
testDf2 = pd.read_csv('C:\\Users\Axim-\Downloads\Yahoo\Webscope_C14\Webscope_C14\Learning to Rank Challenge\ltrc_yahoo.tar\ltrc_yahoo\set2.testParsed.txt')
validDf = pd.read_csv('C:\\Users\Axim-\Downloads\Yahoo\Webscope_C14\Webscope_C14\Learning to Rank Challenge\ltrc_yahoo.tar\ltrc_yahoo\set1.validParsed.txt')
t1 = time.time()
total1 = t1-t0
print("Read done: " + str(total1))

results = [0.0]
roundNr = 6

while roundNr < 7:



    trainTemp = np.array(trainDf)
    testTemp = np.array(testDf)
    testTemp2 = np.array(testDf2)
    validTemp = np.array(validDf)
    trainNP = trainTemp.astype(float)
    testNP = testTemp.astype(float)
    testNP2 = testTemp2.astype(float)
    validNP = validTemp.astype(float)

    trainX = trainNP[:, 1:]
    trainY = trainNP[:, 0]
    testX = testNP[:, 1:]
    testX2 = testNP2[:, 1:]
    testY = testNP[:, 0]
    testY2 = testNP2[:, 0]
    validX = validNP[:, 1:]
    validY = validNP[:, 0]

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
    dtest2 = xgb.DMatrix(testX2, label=testY2)
    dvalid = xgb.DMatrix(validX, label=validY)
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
        'objective': 'multi:softmax',
        'num_class': 5,
        #'objective': 'binary:logistic',
        'tree_method': 'exact',
        'eval_metric': 'merror',
        'early_stopping_rounds': 10,
        'subsample': 0.5,
        #'tree_method': 'auto'
    }
    evallist = [(dvalid, 'eval'), (dtrain, 'train')]
    num_round = 250



    #print("Starting Training: " + str(total4))
    plst = param.items()
    bst = xgb.train(param, dtrain, num_round)

    t5 = time.time()
    #print("End Training: " + str(total5))

    bst.dump_model('dump.raw.txt', 'featmap.txt')

    t6 = time.time()
    total6 = t6-t5
    #print("Dump model: " + str(total6))

    preds = bst.predict(dtest)
    preds2 = bst.predict(dtest2)
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

    ndcgScore = ndcg_score(testY, preds)
    ndcgScore2 = ndcg_score(testY2, preds2)

    print("Score - NDCG@10 Set1:" + str(ndcgScore) + "    Set2: " + str(ndcgScore2))

    t7 = time.time()
    total7 = t7-t6
    #print("Prediction: " + str(total7))

    correctPred = []
    for e in testY:
        correctPred.append(float(e))

    formattedPreds = best_preds

    #xgb.plot_importance(bst)

    #print("Result: " + str(calculateAccuracy(correctPred, formattedPreds)))

    roundNr += 1

    joblib.dump(bst, 'bst_model.pkl', compress=True)
