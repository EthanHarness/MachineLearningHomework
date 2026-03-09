import copy
from cvxopt import matrix, solvers

class SVM_Slack_Part1:
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
        
    def constructPMatrix(self, hyperParam):
        matSize = len(self.trData) + self.wLength + self.bLength
        PList = [[0.0]*matSize for _ in range(matSize)] #Produces a sizeP by sizeP matrix of all zeros
    
        #Sets w coefficiants
        for x in range(self.wLength):
            PList[x][x] = 1.0
        
        #Sets slack coefficiants
        for x in range(self.wLength+self.bLength, matSize):
            PList[x][x] = hyperParam*2.0
            
        return PList
    
    def constructQMatrix(self):
        matSize = len(self.trData) + self.wLength + self.bLength
        return [0.0]*matSize
    
    def constructHMatrix(self):
        return [-1.0]*(len(self.trData))
    
    def constructGMatrix(self):
        matSize = len(self.trData) + self.wLength + self.bLength
        columnVectors = []
        
        for x in range(self.wLength):
            col = [(-1.0*self.trLabel[y]*self.trData[y][x]) for y in range(len(self.trData))]
            columnVectors.append(col)
        
        columnVectors.append([-1.0*x for x in self.trLabel])
        
        for x in range(self.wLength+self.bLength, matSize):
            col = [0.0]*len(self.trData)
            col[x-(self.wLength+self.bLength)] = -1.0
            columnVectors.append(col)
            
        return columnVectors

    def solveWithHyperParam(self, hyperParam):
        pMat = matrix(self.constructPMatrix(hyperParam))
        qMat = matrix(self.constructQMatrix())
        hMat = matrix(self.constructHMatrix())
        gMat = matrix(self.constructGMatrix())
        
        return solvers.qp(pMat, qMat, gMat, hMat)
    
    @staticmethod
    def compareOverHyperParams(trainData, trainLabel, validationData, validationLabel, testData, testLabel, hyperList, computeAccuracy):
        testAccuracyList = {}
        validAccuracyList = {}
        trainer = SVM_Slack_Part1(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
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
        trainer = SVM_Slack_Part1(trainData, trainLabel, [], [], testData, testLabel)
        sol = trainer.solveWithHyperParam(bestHyperParam)

        return computeAccuracy(sol, trainer.tsData, trainer.tsLabel)