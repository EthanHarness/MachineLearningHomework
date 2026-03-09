import copy
from cvxopt import matrix, solvers

class SVM_Slack_Part2:
    def __init__(self, trData, trLabel, vData, vLabel, tsData, tsLabel):
        self.trData = trData
        self.trLabel = trLabel
        self.vData = vData
        self.vLabel = vLabel
        self.tsData = tsData
        self.tsLabel = tsLabel

        assert len(trData[0]) == 10, "Input Data wrong size"
        
        self.wLength = len(trData[0])
        self.bLength = 1
        self.tLength = 1

    def constructPMatrix(self):
        matSize = len(self.trData) + self.wLength + self.bLength + self.tLength
        PList = [[0.0]*matSize for _ in range(matSize)] 
    
        for x in range(self.wLength):
            PList[x][x] = 1.0

        return PList
    
    def constructQMatrix(self, hyperParam):
        qMat = [0.0]*(self.wLength + self.bLength + self.tLength + len(self.trData))
        qMat[-1] = hyperParam
        return qMat
    
    def constructHMatrix(self):
        matSize = len(self.trData)
        return [-1.0]*matSize + [0.0]*(matSize*2)
    
    def constructGMatrix(self):
        dataPoints = len(self.trData)
        createCol = lambda: [0.0]*(dataPoints*3)
        columnVectors = []
        
        #First 10 cols with 3K elements
        for i in range(self.wLength):
            col = createCol()
            for j in range(dataPoints):
                col[j] = (-1)*self.trLabel[j]*self.trData[j][i]
            columnVectors.append(col)
        
        #11th col with 3k elements      
        col = createCol()
        for i in range(dataPoints):
            col[i] = -1*self.trLabel[i]
        columnVectors.append(col)
            
        #Cols 11 through K-1 with 3k elements
        for offset in range(dataPoints):
            col = createCol()
            col[offset] = -1.0
            col[offset+dataPoints] = -1.0
            col[offset+dataPoints+dataPoints] = 1.0
            columnVectors.append(col)
            
        finalCol = [0.0]*(dataPoints*2) + [-1.0]*(dataPoints)
        columnVectors.append(finalCol)
        
        return columnVectors
    
    def solveWithHyperParam(self, hyperParam):
        pMat = matrix(self.constructPMatrix())
        qMat = matrix(self.constructQMatrix(hyperParam))
        hMat = matrix(self.constructHMatrix())
        gMat = matrix(self.constructGMatrix())
        
        return solvers.qp(pMat, qMat, gMat, hMat)
    
    @staticmethod
    def compareOverHyperParams(trainData, trainLabel, validationData, validationLabel, testData, testLabel, hyperList, computeAccuracy):
        testAccuracyList = {}
        validAccuracyList = {}
        trainer = SVM_Slack_Part2(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
        for x in hyperList:
            sol = trainer.solveWithHyperParam(x)
            testAccuracyList[x] = computeAccuracy(sol, trainer.tsData, trainer.tsLabel)
            validAccuracyList[x] = computeAccuracy(sol, trainer.vData, trainer.vLabel)
        return testAccuracyList,validAccuracyList
    
    @staticmethod
    def combinedTrainValidModel(trainData, trainLabel, validationData, validationLabel, testData, testLabel, bestHyperParam, computeAccuracy):
        validDCopy = copy.deepcopy(validationData)
        validLCopy = copy.deepcopy(validationLabel)
        trainData = trainData + validDCopy
        trainLabel = trainLabel + validLCopy
        trainer = SVM_Slack_Part2(trainData, trainLabel, [], [], testData, testLabel)
        sol = trainer.solveWithHyperParam(bestHyperParam)

        return computeAccuracy(sol, trainer.tsData, trainer.tsLabel)