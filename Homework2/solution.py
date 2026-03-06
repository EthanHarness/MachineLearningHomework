import copy
import math
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
    def compareOverHyperParams(trainData, trainLabel, validationData, validationLabel, testData, testLabel):
        testAccuracyList = []
        validAccuracyList = []
        trainer = SVM_Slack_Normal(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
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
    
class SVM_Slack_GaussianKernel:
    def __init__(self, trData, trLabel, vData, vLabel, tsData, tsLabel):
        self.trData = trData
        self.trLabel = trLabel
        self.vData = vData
        self.vLabel = vLabel
        self.tsData = tsData
        self.tsLabel = tsLabel
        
        self.wLength = len(trData[0])
        self.bLength = 1

    def constructPMatrix(self, sigma):
        matSize = len(self.trData)
        PList = [[0.0]*matSize for _ in range(matSize)] 
    
        for x in range(matSize):
            PList[x][x] = self.trLabel[x] * self.trLabel[x]
            for y in range(x+1, matSize):
                kVal = SVM_Slack_GaussianKernel.computeGaussianKernel(self.trData[y], self.trData[x], sigma)
                val = self.trLabel[y]*self.trLabel[x]*kVal
                
                PList[x][y] = val
                PList[y][x] = val

        return PList
    
    def constructQMatrix(self):
        matSize = len(self.trData)
        return [-1.0]*matSize
    
    def constructHMatrix(self, hyperParam):
        matSize = len(self.trData)
        return [hyperParam]*matSize + [0.0]*matSize
    
    def constructGMatrix(self):
        width = len(self.trData)
        leng = len(self.trData)*2
        columnVectors = [[0.0]*leng for _ in range(width)]

        for x in range(width):
            columnVectors[x][x] = 1.0
            columnVectors[x][x+len(self.trData)] = -1.0

        return columnVectors

    def constructAMatrix(self):
        return [[float(y)] for y in self.trLabel]
    
    def solveWithHyperParamAndSigma(self, hyperParam, sigma):
        pMat = matrix(self.constructPMatrix(sigma))
        qMat = matrix(self.constructQMatrix())
        hMat = matrix(self.constructHMatrix(hyperParam))
        gMat = matrix(self.constructGMatrix())
        aMat = matrix(self.constructAMatrix())
        bMat = matrix([0.0])
        
        return solvers.qp(pMat, qMat, gMat, hMat, aMat, bMat)
        

    @staticmethod
    def computeGaussianKernel(X1, X2, sigma):
        return math.exp((sum([((x-y)**2) for x,y in zip(X1,X2)])*(-1)) / (2*sigma*sigma))
    
    @staticmethod
    def compareOverHyperParamsAndSigmas(trainData, trainLabel, validationData, validationLabel, testData, testLabel):
        testAccuracyDict = {x: [] for x in hyperList}
        validAccuracyDict = {x: [] for x in hyperList}
        trainer = SVM_Slack_GaussianKernel(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
        for x in hyperList:
            for y in sigmaList:
                sol = trainer.solveWithHyperParamAndSigma(x, y)
                testAccuracyDict[x].append(computeKernelAccuracy(sol, trainer.trData, trainer.trLabel, trainer.tsData, trainer.tsLabel, y))
                validAccuracyDict[x].append(computeKernelAccuracy(sol, trainer.trData, trainer.trLabel, trainer.vData, trainer.vLabel, y))
        return testAccuracyDict,validAccuracyDict


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
    wVector = solution['x'][0:len(data[0])]
    bias = solution['x'][len(data[0])]
    
    classes = [
        y * (sum([xi * wi for xi, wi in zip(x, wVector)]) + bias) 
        for x, y in zip(data, label)
    ]
    return sum([1 if x > 0 else 0 for x in classes]) / len(data)


def computeKernelAccuracy(sol, trData, trLabel, testData, testLabel, sigma):
    alphas = list(sol['x'])
    
    sv_indices = [i for i, a in enumerate(alphas) if a > 1e-5]
    
    if not sv_indices:
        return 0.0 
    
    s = sv_indices[0]
    
    kernel_sum_s = 0.0
    for i in sv_indices:
        k_val = SVM_Slack_GaussianKernel.computeGaussianKernel(trData[i], trData[s], sigma)
        kernel_sum_s += alphas[i] * trLabel[i] * k_val
        
    bias = trLabel[s] - kernel_sum_s

    correct_count = 0
    for x_test, y_test in zip(testData, testLabel):
        f_x = 0.0
        for i in sv_indices:
            k_val = SVM_Slack_GaussianKernel.computeGaussianKernel(trData[i], x_test, sigma)
            f_x += alphas[i] * trLabel[i] * k_val
        
        prediction = f_x + bias
        
        if (prediction > 0 and y_test > 0) or (prediction < 0 and y_test < 0):
            correct_count += 1
            
    return correct_count / len(testData)


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

    """testAc, validAc = SVM_Slack_Normal.compareOverHyperParams(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel
    )
    print(testAc, validAc)"""
    """print("Accuracy: ", testAc, validAc)

    bestHyper = findBestHyperparam(validAc)
    print(f"Best Hyperparam {bestHyper}:{hyperList[bestHyper]} ")

    accuracy = SVM_Slack_Normal.combinedTrainValidModel(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, bestHyper
    )
    print("Accuracy of New Model: ", accuracy)"""

    testAc, validAc = SVM_Slack_GaussianKernel.compareOverHyperParamsAndSigmas(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel
    )
    print(testAc, validAc)


    
    
    
    

if __name__ == "__main__":
    main()