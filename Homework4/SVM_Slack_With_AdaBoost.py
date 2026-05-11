from __future__ import annotations
from typing import Any, List
from cvxopt import matrix, solvers # type: ignore
import math

solvers.options['show_progress'] = False # type: ignore

class SVM_Slack_With_AdaBoost:
    def __init__(self, trData: List[List[float]], trLabel: List[float]) -> None:
        self.trData: List[List[float]] = trData
        self.trLabel: List[float] = trLabel

        self.trainingDataWeights: List[float] = [1/len(trData) for _ in range(len(trData))]
        
        assert len(trData[0]) == 22, "Input Data wrong size"

        self.wLength: int = len(trData[0])
        self.bLength: int = 1

    def constructPMatrix(self) -> List[List[float]]:
        matSize: int = len(self.trData) + self.wLength + self.bLength
        PList: List[List[float]] = [[0.0]*matSize for _ in range(matSize)] #Produces a sizeP by sizeP matrix of all zeros
    
        #Sets w coefficiants
        for x in range(self.wLength):
            PList[x][x] = 1.0
        
        #Sets slack coefficiants
        for x in range(self.wLength+self.bLength, matSize):
            PList[x][x] = 0.0
            
        return PList
    
    def constructQMatrix(self, hyperParam: float) -> List[float]:
        matSize: int = len(self.trData) + self.wLength + self.bLength
        slackOffset: int = self.wLength + self.bLength
        return [0.0 if x < slackOffset else hyperParam*self.trainingDataWeights[x-slackOffset] for x in range(matSize)]
    
    def constructHMatrix(self) -> List[float]:
        matSize: int = len(self.trData)*2
        return [-1.0 if x < len(self.trData) else 0.0 for x in range(matSize)]
    
    def constructGMatrix(self) -> List[List[float]]:
        colSize: int = len(self.trData)*2
        N: int = len(self.trData)
        columnVectors: List[List[float]] = []
        
        #Constructs first 22 cols of G
        for x in range(self.wLength):
            col: List[float] = [0.0]*colSize
            for y in range(N):
                col[y] = -1*self.trLabel[y]*self.trData[y][x]
            columnVectors.append(col)

        #Constructs the bias column of G
        columnVectors.append([-1.0*self.trLabel[y] if y < N else 0.0 for y in range(colSize)])

        #Constructs the slack cols of G
        for x in range(0, N):
            col: List[float] = [0.0]*colSize
            col[x] = -1.0
            col[N+x] = -1.0
            columnVectors.append(col)

        return columnVectors 
    
    def solveWithHyperParam(self, hyperParam: float) -> dict[str, Any]:
        pMat = matrix(self.constructPMatrix()) # type: ignore
        qMat = matrix(self.constructQMatrix(hyperParam)) # type: ignore
        hMat = matrix(self.constructHMatrix()) # type: ignore
        gMat = matrix(self.constructGMatrix()) # type: ignore
        
        return solvers.qp(pMat, qMat, gMat, hMat) # type: ignore
    
    def adjustWeights(self, epsilon: float, alpha: float, solution: dict[str, Any]) -> None:
        wVector: List[float] = solution['x'][0:len(self.trData[0])]
        bias: float = solution['x'][len(self.trData[0])]
        newDataWeights: List[float] = [0.0] * len(self.trData)
        normalizationConstant: float = 2*math.sqrt(epsilon*(1-epsilon))

        for x in range(len(newDataWeights)):
            prevWeight: float = self.trainingDataWeights[x]
            modelPrediction = 1 if (sum(xi*wi for xi, wi in zip(self.trData[x], wVector)) + bias) >= 0 else -1
            expTerm: float = math.exp(-1*self.trLabel[x]*modelPrediction*alpha)
            newDataWeights[x] = (prevWeight*expTerm) / normalizationConstant

        self.trainingDataWeights = newDataWeights
    
    @staticmethod
    def computeWeightedError(solution: (dict[str, Any] | Any), data: List[List[float]], label: List[float], trainer: SVM_Slack_With_AdaBoost) -> float:
        wVector: List[float] = solution['x'][0:len(data[0])]
        bias: float = solution['x'][len(data[0])]

        assert len(wVector) == 22, "Size mismatch"
        
        classes: List[float] = [
            y * (sum([xi * wi for xi, wi in zip(x, wVector)]) + bias) 
            for x, y in zip(data, label)
        ]

        return sum([0 if x > 0 else trainer.trainingDataWeights[index] for index,x in enumerate(classes)])

    @staticmethod
    def computeAlpha(weightedError: float) -> float:
        return .5*math.log((1-weightedError)/weightedError)
    
    @staticmethod
    def runAdaBoostSVMAlgorithm(trainData: List[List[float]], trainLabel: List[float]) \
        -> List[tuple[float, float, List[float], float]]:

        hyperParameterList: List[float] = [.001, .01, .1, 1.0, 10.0, 100.0, 1000.0]
        adaBoostIterations = 10
        trainer = SVM_Slack_With_AdaBoost(trainData, trainLabel) 
        adaBoostDataList: List[tuple[float, float, List[float], float]] = [] #Holds 10 tuples of (eps, alpha, w, b) for each iteration

        for _ in range(adaBoostIterations):
            bestModel: dict[str, Any] 
            bestError: float|None = None
            for hyperParam in hyperParameterList:
                model = trainer.solveWithHyperParam(hyperParam)
                weightedError = SVM_Slack_With_AdaBoost.computeWeightedError(model, trainData, trainLabel, trainer)
                if bestError is None or weightedError < bestError:
                    bestModel = model
                    bestError = weightedError

            alpha = SVM_Slack_With_AdaBoost.computeAlpha(bestError) # type: ignore
            adaBoostDataList.append((bestError, alpha, bestModel['x'][0:trainer.wLength], bestModel['x'][trainer.wLength])) # type: ignore
            trainer.adjustWeights(bestError, alpha, bestModel) # type: ignore
            
        return adaBoostDataList
            
