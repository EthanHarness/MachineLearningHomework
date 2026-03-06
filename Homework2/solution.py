import copy
from cvxopt import matrix, solvers

hyperList = [0.01,0.1,1.0,10.0,100.0,1000.0,10000.0,100000.0]
sigmaList = [.001,.01,.1,1,10,100]
W_LENGTH = 10
B_LENGTH = 1

class SVM_Slack_Normal:
    def __init__(self, trData, trLabel, vData, vLabel, tsData, tsLabel):
        self.trData = trData
        self.trLabel = trLabel
        self.vData = vData
        self.vLabel = vLabel
        self.tsData = tsData
        self.tsLabel = tsLabel
        
        self.wLength = len(trData[0])
        self.bLength = 1

    def constructPMatrix(self):
        matSize = len(self.trData) + self.wLength + self.bLength
        PList = [[0.0]*matSize for _ in range(matSize)] #Produces a sizeP by sizeP matrix of all zeros
    
        #Sets w coefficiants
        for x in range(W_LENGTH):
            PList[x][x] = 1.0
        
        #Sets slack coefficiants
        for x in range(W_LENGTH+B_LENGTH, matSize):
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
        for x in range(W_LENGTH):
            col = [0.0]*colSize
            for y in range(N):
                col[y] = -1*self.trLabel[y]*self.trData[y][x]
            columnVectors.append(col)

        #Constructs the bias column of G
        columnVectors.append([-1.0 if x < N else 0.0 for x in range(colSize)])

        #Constructs the slack cols of G
        for x in range(0, N-W_LENGTH-B_LENGTH):
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
    def compareOverHyperParams(trainData, trainLabel, validationData, validationLabel, testData, testLabel):
        testAccuracyList = []
        validAccuracyList = []
        trainer = SVM_Slack_Part1(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
        for x in hyperList:
            sol = trainer.solveWithHyperParam(x)
            testAccuracyList.append(computeAccuracy(sol, trainer.tsData, trainer.tsLabel))
            validAccuracyList.append(computeAccuracy(sol, trainer.vData, trainer.vLabel))
        return testAccuracyList,validAccuracyList
    
    @staticmethod
    def combinedTrainValidModel(trainData, trainLabel, validationData, validationLabel, testData, testLabel, bestHyperParamIndex):
        validDCopy = copy.deepcopy(validationData)
        validLCopy = copy.deepcopy(validationLabel)
        trainData = trainData + validDCopy
        trainLabel = trainLabel + validLCopy
        trainer = SVM_Slack_Part1(trainData, trainLabel, [], [], testData, testLabel)
        sol = trainer.solveWithHyperParam(hyperList[bestHyperParamIndex])

        return computeAccuracy(sol, trainer.tsData, trainer.tsLabel)
    
    


class SVM_Slack_Part1:
    def __init__(self, trData, trLabel, vData, vLabel, tsData, tsLabel):
        self.trData = trData
        self.trLabel = trLabel
        self.vData = vData
        self.vLabel = vLabel
        self.tsData = tsData
        self.tsLabel = tsLabel
        
        self.wLength = len(trData[0])
        self.bLength = 1
        
    def constructPMatrix(self, hyperParam):
        matSize = len(self.trData) + self.wLength + self.bLength
        PList = [[0.0]*matSize for _ in range(matSize)] #Produces a sizeP by sizeP matrix of all zeros
    
        #Sets w coefficiants
        for x in range(W_LENGTH):
            PList[x][x] = 1.0
        
        #Sets slack coefficiants
        for x in range(W_LENGTH+B_LENGTH, matSize):
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
            col[x-(W_LENGTH+B_LENGTH)] = -1.0
            columnVectors.append(col)
            
        return columnVectors

    def solveWithHyperParam(self, hyperParam):
        pMat = matrix(self.constructPMatrix(hyperParam))
        qMat = matrix(self.constructQMatrix())
        hMat = matrix(self.constructHMatrix())
        gMat = matrix(self.constructGMatrix())
        
        return solvers.qp(pMat, qMat, gMat, hMat)
    
    @staticmethod
    def compareOverHyperParams(trainData, trainLabel, validationData, validationLabel, testData, testLabel):
        testAccuracyList = []
        validAccuracyList = []
        trainer = SVM_Slack_Part1(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
        for x in hyperList:
            sol = trainer.solveWithHyperParam(x)
            testAccuracyList.append(computeAccuracy(sol, trainer.tsData, trainer.tsLabel))
            validAccuracyList.append(computeAccuracy(sol, trainer.vData, trainer.vLabel))
        return testAccuracyList,validAccuracyList
    
        
def computeAccuracy(solution, data, label):
    wVector = solution['x'][0:10]
    bias = solution['x'][10]
    
    classes = [
        y * (sum([xi * wi for xi, wi in zip(x, wVector)]) + bias) 
        for x, y in zip(data, label)
    ]
    return sum([1 if x > 0 else 0 for x in classes]) / len(data)    

def findBestHyperparam(validAc):
    return validAc.index(max(validAc))


def getInputData():
    inputData = []
    inputDataLabels = []
    with open("./magic.data", "r") as file:
        for line in file:
            valArr = "".join(line.split())
            valArr = [float(i) for i in valArr.split(',')]
            valArr[-1] = int(valArr[-1]) if int(valArr[-1]) == 1 else -1 #Changes 0 classifications to -1
            inputData.append(valArr[:-1])
            inputDataLabels.append(valArr[-1])
            
    return (inputData, inputDataLabels)


def main():
    data, labels = getInputData()
    trainData, trainLabel = data[0:1800], labels[0:1800]
    validationData, validationLabel = data[1800:2400], labels[1800:2400]
    testData, testLabel = data[2400:], labels[2400:]

    """testAc, validAc = SVM_Slack_Part1.compareOverHyperParams(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel
    )"""

    testAc, validAc = SVM_Slack_Normal.compareOverHyperParams(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel
    )
    print("Accuracy: ", testAc, validAc)

    bestHyper = findBestHyperparam(validAc)
    print(f"Best Hyperparam {bestHyper}:{hyperList[bestHyper]} ")

    accuracy = SVM_Slack_Normal.combinedTrainValidModel(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, bestHyper
    )
    print("Accuracy of New Model: ", accuracy)

    
    
    
    

if __name__ == "__main__":
    main()