import copy
from cvxopt import matrix, solvers

# Class for problem2 part 1
class SVM_Slack_Normal:
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

    def constructPMatrix(self):
        matSize = len(self.trData) + self.wLength + self.bLength
        PList = [[0.0]*matSize for _ in range(matSize)] #Produces a sizeP by sizeP matrix of all zeros
    
        #Sets w coefficiants
        for x in range(self.wLength):
            PList[x][x] = 1.0
        
        #Sets slack coefficiants
        for x in range(self.wLength+self.bLength, matSize):
            PList[x][x] = 0.0
            
        return PList
    
    def constructQMatrix(self, hyperParam):
        matSize = len(self.trData) + self.wLength + self.bLength
        return [0.0 if x < 11 else hyperParam for x in range(matSize)]
    
    def constructHMatrix(self):
        matSize = len(self.trData)*2
        return [-1.0 if x < len(self.trData) else 0.0 for x in range(matSize)]
    
    def constructGMatrix(self):
        colSize = len(self.trData)*2
        N = len(self.trData)
        columnVectors = []
        
        #Constructs first 10 cols of G
        for x in range(self.wLength):
            col = [0.0]*colSize
            for y in range(N):
                col[y] = -1*self.trLabel[y]*self.trData[y][x]
            columnVectors.append(col)

        #Constructs the bias column of G
        columnVectors.append([-1.0*self.trLabel[y] if y < N else 0.0 for y in range(colSize)])

        #Constructs the slack cols of G
        for x in range(0, N):
            col = [0.0]*colSize
            col[x] = -1.0
            col[N+x] = -1.0
            columnVectors.append(col)

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
        trainer = SVM_Slack_Normal(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
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
        trainer = SVM_Slack_Normal(trainData, trainLabel, [], [], testData, testLabel)
        sol = trainer.solveWithHyperParam(bestHyperParam)

        return computeAccuracy(sol, trainer.tsData, trainer.tsLabel)