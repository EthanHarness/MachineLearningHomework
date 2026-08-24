from typing import List, Tuple
from Logistic_Regression import LogisticRegression
from Logistic_Regression_L2 import LogisticRegressionL2
from Logistic_Regression_L1 import LogisticRegressionL1
from Logistic_SVM import SVM_Normal, generateTestData
import matplotlib.pyplot as plt
import numpy as np
import math

def getInput(dataPath: str) -> Tuple[List[List[float]], List[int]]:
    print(f"Loading from {dataPath}")
    
    dataSet: List[List[float]] = []
    labelSet: List[int] = []    
    with open(dataPath, 'r') as file:
        for line in file:
            valArr: List[float] = [float(i) for i in "".join(line.split()).split(',')]
            labelSet.append(-1 if valArr[-1] == 1 else 1)
            dataSet.append(valArr[:-1])

    return dataSet, labelSet

def problem1Code(trData, trLabel, lr: float = .01, iter: int = 100):
    print(f"Part 1: Logistic Regression Nomral with alpha={lr} running for {iter} iterations")
    md = LogisticRegression(trData, trLabel, lr, iter)
    print(f"Final Accuracy on training set: {md.computeAccuracy()}")
    print("\n\n")

def problem2Code(trData, trLabel, vsData, vsLabel, tsData, tsLabel, lr: float = .01, iter: int = 100, hypers: List[float]|None = None):
    if hypers == None: hypers = [.001, .01, .1, 1, 10, 100, 1000]

    print(f"Part 2: Logistic Regression with L2 penalty with alpha={lr} running for {iter} iterations")
    print("(Hyper Parameter Value) -> (Validation Accuracy on model trained with specific hyper parameter)")
    
    hyperMdDict = LogisticRegressionL2.trainModelOverHyperParameters(trData, trLabel, vsData, vsLabel, lr, iter, hypers)
    bestAcc = 0
    mdKey = None
    for hyper,(acc, _) in hyperMdDict.items():
        print(f"{hyper} -> {acc}")
        if bestAcc == 0 or acc > bestAcc:
            bestAcc = acc
            mdKey = hyper
            
    tAcc = LogisticRegressionL2.computeAccuracyForHypers(tsData, tsLabel, hyperMdDict[mdKey][1])

    print(f"Regularization Constant: {mdKey}")
    print(f"Best Model Weights: {hyperMdDict[mdKey][1].weights}")
    print(f"Best Model Bias: {hyperMdDict[mdKey][1].bias}")
    print(f"Best Model Test Accuracy: {tAcc}")
    print("\n\n")

def problem3Code(trData, trLabel, vsData, vsLabel, tsData, tsLabel, lr: float = .01, iter: int = 100, hypers: List[float]|None = None):
    if hypers == None: hypers = [.001, .01, .1, 1, 10, 100, 1000]

    print(f"Part 3: Logistic Regression with L1 penalty with alpha={lr} running for {iter} iterations")
    print("(Hyper Parameter Value) -> (Validation Accuracy on model trained with specific hyper parameter)")
    
    hyperMdDict = LogisticRegressionL1.trainModelOverHyperParameters(trData, trLabel, vsData, vsLabel, lr, iter, hypers)
    bestAcc = 0
    mdKey = None
    for hyper,(acc, _) in hyperMdDict.items():
        print(f"{hyper} -> {acc}")
        if bestAcc == 0 or acc > bestAcc:
            bestAcc = acc
            mdKey = hyper
            
    tAcc = LogisticRegressionL1.computeAccuracyForHypers(tsData, tsLabel, hyperMdDict[mdKey][1])

    print(f"Regularization Constant: {mdKey}")
    print(f"Best Model Weights: {hyperMdDict[mdKey][1].weights}")
    print(f"Best Model Bias: {hyperMdDict[mdKey][1].bias}")
    print(f"Best Model Test Accuracy: {tAcc}")
    print("\n\n")

def part4Answer():
    print("In general L1 Regularization produces sparser weights. ")

def part5(numData: int = 100, validProportion: float = .1, lr: float = .1, iter: int = 100, hypers: List[float]|None = None):
    if hypers == None: hypers = [.001, .01, .1, 1, 10, 100, 1000]

    datas, coef = generateTestData(numData)    
    data: List[List[float]] = []
    labelD: List[int] = []
    for (x1,x2,label) in datas:
        data.append([x1, x2])
        labelD.append(label)

    trainingNum = numData - math.floor(numData*validProportion)
    trData = data[0:trainingNum]
    trLabel = labelD[0:trainingNum]
    vsData = data[trainingNum:]
    vsLabel = labelD[trainingNum:]

    md = SVM_Normal(trData, trLabel, hypers[0])
    dict = md.solveWithHyperParam()

    mdsDict = LogisticRegressionL2.trainModelOverHyperParameters(trData, trLabel, vsData, vsLabel, lr, iter, hypers)
    bestAcc = 0
    mdKey = None
    for hyper,(acc, _) in mdsDict.items():
        print(f"{hyper} -> {acc}")
        if bestAcc == 0 or acc > bestAcc:
            bestAcc = acc
            mdKey = hyper

    createPlots(trData, dict, mdsDict[mdKey][1])


def createPlots(trData, SVMModel, L2Model):
    def SVM_Plot_Helper(x1, model) -> float:
        w1 = model['x'][0]
        w2 = model['x'][1]
        bias = model['x'][2]
        return (-bias - (x1*w1))/w2
    
    def L2_Plot_Helper(x1, model) -> float:
        w1 = model.weights[0]
        w2 = model.weights[1]
        bias = model.bias
        return (-bias - (x1*w1))/w2

    xData = []
    yData = []
    SVM_Data = []
    L2_Data = []
    for vals in trData:
        x = vals[0]
        y = vals[1]

        xData.append(x)
        yData.append(y)
        SVM_Data.append(SVM_Plot_Helper(x, SVMModel))
        L2_Data.append(L2_Plot_Helper(x, L2Model))
    
    xData = np.array(xData)
    yData = np.array(yData)
    SVM_Data = np.array(SVM_Data)
    L2_Data = np.array(L2_Data)

    plt.scatter(xData, yData)
    plt.plot(xData, SVM_Data)
    plt.plot(xData, L2_Data)
    plt.show()


def main():
    trData, trLabel = getInput('./sonar_train.data')
    tsData, tsLabel = getInput('./sonar_test.data')
    vsData, vsLabel = getInput('./sonar_valid.data')
    print("\n\n")

    hypers = [.001, .01, .1, 1, 10, 100, 1000]

    #problem1Code(trData, trLabel, .01, 3000)
    #problem2Code(trData, trLabel, vsData, vsLabel, tsData, tsLabel, .01, 3000, hypers)
    #problem3Code(trData, trLabel, vsData, vsLabel, tsData, tsLabel, .01, 3000, hypers)
    #part4Answer()
    part5()


if __name__ == "__main__":
    main()