import copy
from typing import Any, Callable, List
from cvxopt import matrix, solvers

class SVM_Slack_Part2:
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
        self.tLength = 1

    def constructPMatrix(self) -> List[List[float]]:
        matSize: int = len(self.trData) + self.wLength + self.bLength + self.tLength
        PList: List[List[float]] = [[0.0]*matSize for _ in range(matSize)] 
    
        for x in range(self.wLength):
            PList[x][x] = 1.0

        return PList
    
    def constructQMatrix(self, hyperParam: float) -> List[float]:
        qMat: List[float] = [0.0]*(self.wLength + self.bLength + self.tLength + len(self.trData))
        qMat[-1] = hyperParam
        return qMat
    
    def constructHMatrix(self) -> List[float]:
        matSize: int = len(self.trData)
        return [-1.0]*matSize + [0.0]*(matSize*2)
    
    def constructGMatrix(self) -> List[List[float]]:
        dataPoints: int = len(self.trData)
        createCol: Callable[[], List[float]] = lambda: [0.0]*(dataPoints*3)
        columnVectors: List[List[float]] = []
        
        #First 10 cols with 3K elements
        for i in range(self.wLength):
            col: List[float] = createCol()
            for j in range(dataPoints):
                col[j] = (-1)*self.trLabel[j]*self.trData[j][i]
            columnVectors.append(col)
        
        #11th col with 3k elements      
        col: List[float] = createCol()
        for i in range(dataPoints):
            col[i] = -1*self.trLabel[i]
        columnVectors.append(col)
            
        #Cols 11 through K-1 with 3k elements
        for offset in range(dataPoints):
            col: List[float] = createCol()
            col[offset] = -1.0
            col[offset+dataPoints] = -1.0
            col[offset+dataPoints+dataPoints] = 1.0
            columnVectors.append(col)
            
        finalCol: List[float] = [0.0]*(dataPoints*2) + [-1.0]*(dataPoints)
        columnVectors.append(finalCol)
        
        return columnVectors
    
    def solveWithHyperParam(self, hyperParam) -> dict[str, Any]:
        pMat = matrix(self.constructPMatrix())
        qMat = matrix(self.constructQMatrix(hyperParam))
        hMat = matrix(self.constructHMatrix())
        gMat = matrix(self.constructGMatrix())
        
        return solvers.qp(pMat, qMat, gMat, hMat)
    
    @staticmethod
    def compareOverHyperParams(trainData: List[List[float]], trainLabel: List[int], validationData: List[List[float]], validationLabel: List[int], \
                               testData: List[List[float]], testLabel: List[int], hyperList: List[float], computeAccuracy: Callable[[List[List[float]], List[int]], float]) \
                                -> tuple[dict[float, float], dict[float, float]]:
        
        testAccuracyList: dict[float, float] = {}
        validAccuracyList: dict[float, float] = {}
        trainer: SVM_Slack_Part2 = SVM_Slack_Part2(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
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
        trainer: SVM_Slack_Part2 = SVM_Slack_Part2(trainData, trainLabel, [], [], testData, testLabel)
        sol: dict[str, Any] = trainer.solveWithHyperParam(bestHyperParam)

        return computeAccuracy(sol, trainer.tsData, trainer.tsLabel)