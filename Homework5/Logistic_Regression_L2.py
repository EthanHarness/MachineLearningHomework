from __future__ import annotations
import math
from typing import List


class LogisticRegressionL2:
    def __init__(self, tData, tLabel, lr: float = .1, iter: int = 100, lam: float = .01):
        self.trData = tData
        self.trLabel = tLabel

        self.wLength = len(tData[0])
        self.lr = lr
        self.iter = iter
        self.lam = lam

        self.weights: List[float] = ([0]*self.wLength)
        self.bias: float = 0

        self.gradientAscent()

    def innerProdPlusBias(self, xVector: List[float]) -> float:
        sum = 0
        assert len(xVector) == len(self.weights), 'FIX THIS'
        for wVal, xVal in zip(self.weights, xVector):
            sum += (wVal * xVal)
        sum += self.bias
        return sum
    
    def sigmoid(self, val: float):
        if val > 709: return 1
        if val < -709: return 0
        return 1 / (1 + math.exp(val*-1))
    
    def computeBGrad(self):
        sum = 0
        for xVec, label in zip(self.trData, self.trLabel):
            val = self.sigmoid(self.innerProdPlusBias(xVec))
            sum += (((label+1)/2) - val)
        return sum
    
    def computeWGrads(self):
        wGrads: List[float] = [0]*self.wLength
        assert len(self.trData[0]) == self.wLength, 'FIX THIS'

        sum = 0
        for xVec, label in zip(self.trData, self.trLabel):
            y = (label + 1) / 2
            val = self.sigmoid(self.innerProdPlusBias(xVec))
            error = y - val

            for index, xVal in enumerate(xVec):
                wGrads[index] += error*xVal

        for j in range(self.wLength):
            wGrads[j] -= self.lam * self.weights[j]

        return wGrads
    
    def updateParams(self, bGrad: float, wGrads: List[float]):
        self.bias = self.bias + (self.lr * (bGrad))
        newWeights: List[float] = [0.0]*len(wGrads)
        
        assert len(self.weights) == len(wGrads), "Fix this"
        for index,(mWeight, wGrad) in enumerate(zip(self.weights, wGrads)):
            newWeights[index] = mWeight + (self.lr * wGrad)

        self.weights = newWeights

    def computeAccuracy(self):
        count = 0
        for xVec, label in zip(self.trData, self.trLabel):
            pred = self.sigmoid(self.innerProdPlusBias(xVec))
            predLabel = 1 if pred >= 0.5 else -1
            if predLabel == label: count += 1
        return count / len(self.trLabel)

    def gradientAscent(self):
        for it in range(self.iter):
            bGrad = self.computeBGrad()
            wGrads = self.computeWGrads()
            self.updateParams(bGrad, wGrads)


    @staticmethod
    def trainModelOverHyperParameters(trData: List[List[float]], trLabel: List[int], \
            tsData: List[List[float]], tsLabel: List[int], lr: float, iterations: int, hypers: List[float]) -> dict[float, tuple[float, LogisticRegressionL2|None]]:
        
        hyperToAccDict: dict[float, tuple[float, LogisticRegressionL2|None]] = {x: (0, None) for x in hypers}
        for hyper in hypers: 
            md = LogisticRegressionL2(trData, trLabel, lr, iterations, hyper)
            acc = LogisticRegressionL2.computeAccuracyForHypers(tsData, tsLabel, md)
            hyperToAccDict[hyper] = (acc, md)

        return hyperToAccDict

    @staticmethod
    def computeAccuracyForHypers(tsData, tsLabel, model: LogisticRegressionL2) -> float:
        count = 0
        for xVec, xLabel in zip(tsData, tsLabel): 
            pred = model.sigmoid(model.innerProdPlusBias(xVec))
            predLabel = 1 if pred >= 0.5 else -1
            if predLabel == xLabel: count += 1
        return count / len(tsData)
    
