import copy
from typing import Any, Callable, List
from cvxopt import matrix, solvers

class SVM_Slack_Part1:
    def __init__(self, trData: List[List[float]], trLabel: List[int], vData: List[List[float]], vLabel: List[int], tsData: List[List[float]], tsLabel: List[int]) -> None:
        self.trData: List[List[float]] = trData
        self.trLabel: List[int] = trLabel
        self.vData: List[List[float]] = vData
        self.vLabel: List[int] = vLabel
        self.tsData: List[List[float]] = tsData
        self.tsLabel: List[int] = tsLabel
        
        assert len(trData[0]) == 10, "Input Data wrong size"

        self.wLength: int = len(trData[0])
        self.bLength: int = 1
        
    def constructPMatrix(self, hyperParam: float) -> List[List[float]]:
        matSize: int = len(self.trData) + self.wLength + self.bLength
        PList: List[List[float]] = [[0.0]*matSize for _ in range(matSize)] #Produces a sizeP by sizeP matrix of all zeros
    
        #Sets w coefficiants
        for x in range(self.wLength):
            PList[x][x] = 1.0
        
        #Sets slack coefficiants
        for x in range(self.wLength+self.bLength, matSize):
            PList[x][x] = hyperParam*2.0
            
        return PList
    
    def constructQMatrix(self) -> List[float]:
        matSize: int = len(self.trData) + self.wLength + self.bLength
        return [0.0]*matSize
    
    def constructHMatrix(self) -> List[float]:
        return [-1.0]*(len(self.trData))
    
    def constructGMatrix(self) -> List[List[float]]:
        matSize: int = len(self.trData) + self.wLength + self.bLength
        columnVectors: List[List[float]] = []
        
        for x in range(self.wLength):
            col: List[float] = [(-1.0*self.trLabel[y]*self.trData[y][x]) for y in range(len(self.trData))]
            columnVectors.append(col)
        
        columnVectors.append([-1.0*x for x in self.trLabel])
        
        for x in range(self.wLength+self.bLength, matSize):
            col: List[float] = [0.0]*len(self.trData)
            col[x-(self.wLength+self.bLength)] = -1.0
            columnVectors.append(col)
            
        return columnVectors

    def solveWithHyperParam(self, hyperParam: float) -> dict[str, Any]:
        pMat = matrix(self.constructPMatrix(hyperParam))
        qMat = matrix(self.constructQMatrix())
        hMat = matrix(self.constructHMatrix())
        gMat = matrix(self.constructGMatrix())
        
        return solvers.qp(pMat, qMat, gMat, hMat)
    
    @staticmethod
    def compareOverHyperParams(trainData: List[List[float]], trainLabel: List[int], validationData: List[List[float]], validationLabel: List[int], \
                               testData: List[List[float]], testLabel: List[int], hyperList: List[float], computeAccuracy: Callable[[List[List[float]], List[int]], float]) \
                                -> tuple[dict[float, float], dict[float, float]]:
        
        testAccuracyList: dict[float, float] = {}
        validAccuracyList: dict[float, float] = {}
        trainer: SVM_Slack_Part1 = SVM_Slack_Part1(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
        for x in hyperList:
            sol: dict[str, Any] = trainer.solveWithHyperParam(x)
            testAccuracyList[x] = computeAccuracy(sol, trainer.tsData, trainer.tsLabel)
            validAccuracyList[x] = computeAccuracy(sol, trainer.vData, trainer.vLabel)
        return testAccuracyList,validAccuracyList
    
    @staticmethod
    def combinedTrainValidModel(trainData: List[List[float]], trainLabel: List[int], validationData: List[List[float]], validationLabel: List[int], \
                                testData: List[List[float]], testLabel: List[int], bestHyperParam: float, computeAccuracy: Callable[[List[List[float]], List[int]], float]) -> float:
        
        validDCopy: List[List[float]] = copy.deepcopy(validationData)
        validLCopy: List[int] = copy.deepcopy(validationLabel)
        trainData: List[List[float]] = trainData + validDCopy
        trainLabel: List[int] = trainLabel + validLCopy
        trainer: SVM_Slack_Part1 = SVM_Slack_Part1(trainData, trainLabel, [], [], testData, testLabel)
        sol: dict[str, Any] = trainer.solveWithHyperParam(bestHyperParam)

        return computeAccuracy(sol, trainer.tsData, trainer.tsLabel)