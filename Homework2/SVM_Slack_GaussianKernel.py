import copy
import math
from cvxopt import matrix, solvers

class SVM_Slack_GaussianKernel:
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
    
    #Look Over
    @staticmethod
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
    
    @staticmethod
    def compareOverHyperParamsAndSigmas(trainData, trainLabel, validationData, validationLabel, testData, testLabel, hyperList, sigmaList):
        tempStructure = {x: {y: 0 for y in hyperList} for x in sigmaList}
        testAccuracyDict = copy.deepcopy(tempStructure)
        validAccuracyDict = copy.deepcopy(tempStructure)
        trainer = SVM_Slack_GaussianKernel(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
        for x in sigmaList:
            for y in hyperList:
                sol = trainer.solveWithHyperParamAndSigma(y, x)
                testAccuracyDict[x][y] = SVM_Slack_GaussianKernel.computeKernelAccuracy(sol, trainer.trData, trainer.trLabel, trainer.tsData, trainer.tsLabel, y)
                validAccuracyDict[x][y] = SVM_Slack_GaussianKernel.computeKernelAccuracy(sol, trainer.trData, trainer.trLabel, trainer.vData, trainer.vLabel, y)
        return testAccuracyDict,validAccuracyDict
    
    
    
    @staticmethod
    def combinedTrainValidModel(trainData, trainLabel, validationData, validationLabel, testData, testLabel, sigmaList, bestHyperForSigma):
        validDCopy = copy.deepcopy(validationData)
        validLCopy = copy.deepcopy(validationLabel)
        trainData = trainData + validDCopy
        trainLabel = trainLabel + validLCopy
        trainer = SVM_Slack_GaussianKernel(trainData, trainLabel, [], [], testData, testLabel)
        accuracyForSigmaDict = {}
        for sig in sigmaList:
            hyperParam = bestHyperForSigma[sig]
            sol = trainer.solveWithHyperParamAndSigma(hyperParam, sig)
            accuracyForSigmaDict[sig] = SVM_Slack_GaussianKernel.computeKernelAccuracy(sol, trainer.trData, trainer.trLabel, trainer.tsData, trainer.tsLabel, sig)
        return accuracyForSigmaDict
    
    @staticmethod
    def findBestCForASigma(validDict):
        sigmaToHyperDict = {}
        for keySig, acList in validDict.items():
            bestHyper = max(acList, key=acList.get)
            sigmaToHyperDict[keySig] = bestHyper
        return sigmaToHyperDict