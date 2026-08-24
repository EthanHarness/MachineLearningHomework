import numpy as np
import random
from typing import Any, List
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False # type: ignore

RANGE_CONSTANT = 100
NOISE_CONSTANT = 100

#Generates some arbitrary hyperplane in R2
#Then adds or subtracts some amount of noise to create points
def generateTestData(numPoints):
    rng = np.random.default_rng()
    coefficiants = rng.uniform(-RANGE_CONSTANT, RANGE_CONSTANT, size=3)
    dataPoints = []
    
    for x in range(numPoints):
        for label in [-1, 1]:
            dragAmount = (rng.uniform(5, NOISE_CONSTANT+5)) * label 
            
            x1 = rng.uniform(-RANGE_CONSTANT, RANGE_CONSTANT)
            x2 = ((coefficiants[0]*x1*-1) - coefficiants[2]) / coefficiants[1] 

            dataPoints.append((x1 + dragAmount, x2 + dragAmount, label))
    return (dataPoints, coefficiants)

class SVM_Normal:
    def __init__(self, trData: List[List[float]], trLabel: List[int], hyper: float) -> None:
        self.trData: List[List[float]] = trData
        self.trLabel: List[int] = trLabel

        self.wLength: int = len(trData[0])
        self.bLength: int = 1
        self.hyper = hyper

    def constructPMatrix(self) -> List[List[float]]:
        matSize: int = self.wLength + self.bLength
        PList: List[List[float]] = [[0.0]*matSize for _ in range(matSize)] #Produces a sizeP by sizeP matrix of all zeros
    
        #Sets w coefficiants
        for x in range(self.wLength):
            PList[x][x] = 1.0
            
        return PList
    
    def constructQMatrix(self) -> List[float]:
        return [0.0] * (self.wLength + self.bLength)
    
    def constructHMatrix(self) -> List[float]:
        return [-1.0] * len(self.trData)
    
    def constructGMatrix(self) -> List[List[float]]:
        colSize: int = len(self.trData)
        N: int = len(self.trData)
        columnVectors: List[List[float]] = []
        
        for x in range(self.wLength):
            col: List[float] = [0.0]*colSize
            for y in range(N):
                col[y] = -1*self.trLabel[y]*self.trData[y][x]
            columnVectors.append(col)

        #Constructs the bias column of G
        columnVectors.append([-1.0*self.trLabel[y] if y < N else 0.0 for y in range(colSize)])

        return columnVectors 
    
    def solveWithHyperParam(self) -> dict[str, Any]:
        pMat = matrix(self.constructPMatrix())
        qMat = matrix(self.constructQMatrix())
        hMat = matrix(self.constructHMatrix())
        gMat = matrix(self.constructGMatrix())
        
        return solvers.qp(pMat, qMat, gMat, hMat)
    
    @staticmethod
    def computeAccuracy(modelDict, trData, trLabel):
        w1 = modelDict['x'][0]
        w2 = modelDict['x'][1]
        bias = modelDict['x'][2]

        count = 0
        for x, label in zip(trData, trLabel):
            predict = 1 if ((w1*x[0]) + (w2*x[1]) + bias) > 0 else -1
            if predict == label: count += 1
        return count / len(trData)




        
