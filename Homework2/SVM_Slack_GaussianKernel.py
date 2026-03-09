import copy
import math
from typing import Any, List
from cvxopt import matrix, solvers

class SVM_Slack_GaussianKernel:
    def __init__(self, trData: List[List[float]], trLabel: List[int], vData: List[List[float]], vLabel: List[int], tsData: List[List[float]], tsLabel: List[int]) -> None:
        self.trData: List[List[float]] = trData
        self.trLabel: List[int] = trLabel
        self.vData: List[List[float]] = vData
        self.vLabel: List[int] = vLabel
        self.tsData: List[List[float]] = tsData
        self.tsLabel: List[int] = tsLabel
        
        assert len(trData[0]) == 10, "Input Data wrong size"

        self.distBtwnTrData: List[List[float]] = self.precomputeSquaredDistances(self.trData, self.trData)
        self.distBtwnSVAndTs: List[List[float]] = []
        self.distBtwnSVAndVl: List[List[float]] = []

        self.wLength: int = len(trData[0])
        self.bLength: int = 1

    def precomputeSquaredDistances(self, setA, setB) -> List[List[float]]:
        distanceMat = [[0.0]*len(setB) for _ in range(len(setA))]
        for i in range(len(setA)):
            for j in range(len(setB)):
                distanceCalc = sum([((x-y)**2) for x,y in zip(setA[i],setB[j])])*(-1)
                distanceMat[i][j] = distanceCalc
        return distanceMat

    def constructPMatrix(self, sigma: float) -> List[List[float]]:
        matSize: int = len(self.trData)
        PList: List[List[float]] = [[0.0]*matSize for _ in range(matSize)] 
        denom = 1/(2*sigma*sigma)
    
        for x in range(matSize):
            PList[x][x] = 1.0
            for y in range(x+1, matSize):
                distance = self.distBtwnTrData[x][y]
                kVal: float = math.exp(distance*denom)
                val: float = self.trLabel[y]*self.trLabel[x]*kVal
                
                PList[x][y] = val
                PList[y][x] = val

        return PList
    
    def constructQMatrix(self) -> List[float]:
        matSize: int = len(self.trData)
        return [-1.0]*matSize
    
    def constructHMatrix(self, hyperParam: float) -> List[float]:
        matSize: int = len(self.trData)
        return [hyperParam]*matSize + [0.0]*matSize
    
    def constructGMatrix(self):
        width: int = len(self.trData)
        leng: int = len(self.trData)*2
        columnVectors: List[List[float]] = [[0.0]*leng for _ in range(width)]

        for x in range(width):
            columnVectors[x][x] = 1.0
            columnVectors[x][x+len(self.trData)] = -1.0

        return columnVectors

    def constructAMatrix(self) -> List[List[float]]:
        return [[float(y)] for y in self.trLabel]
    
    def solveWithHyperParamAndSigma(self, hyperParam: float, sigma: float) -> dict[str, Any]:
        pMat = matrix(self.constructPMatrix(sigma))
        qMat = matrix(self.constructQMatrix())
        hMat = matrix(self.constructHMatrix(hyperParam))
        gMat = matrix(self.constructGMatrix())
        aMat = matrix(self.constructAMatrix())
        bMat = matrix([0.0])
        
        return solvers.qp(pMat, qMat, gMat, hMat, aMat, bMat)
        
    
    # mode=true is training and test 
    # mode=false is training and valid
    def computeKernelAccuracy(self, sol, sigma: float, mode: bool) -> List[float]:
        alphas: List[float] = sol['x']
        denom: float = 1/(2*sigma*sigma)
        cache: List[List[float]] = self.distBtwnSVAndTs if mode else self.distBtwnSVAndVl
        needToPrecompute = True if len(cache) == 0 else False

        trData = self.trData
        trLabel = self.trLabel
        testData = self.tsData if mode else self.vData
        testLabel = self.tsLabel if mode else self.vLabel

        assert len(alphas) == len(trData) and len(alphas) == len(trLabel), "Length Mismatch"
        supportVectorIndexes: List[int] = [index for index,val in enumerate(alphas) if val > 1e-5]
        supportVectorInputVals: List[List[float]] = [trData[index] for index,val in enumerate(alphas) if val > 1e-5]
        
        #Finds b vector (b could just be 1 result of a support vector, but we are making it be an average to be more accurate)
        bValue: float = 0.0
        for s in supportVectorIndexes:
            kernel_sum = 0.0
            for i in supportVectorIndexes:
                distance = self.distBtwnTrData[i][s]
                kernel_sum += (alphas[i] * trLabel[i] * math.exp(distance*denom))

            bValue += (trLabel[s] - kernel_sum)
        bValue = bValue / len(supportVectorIndexes)

        if needToPrecompute:
            if mode: 
                self.distBtwnSVAndTs = self.precomputeSquaredDistances(supportVectorInputVals, testData)
                cache = self.distBtwnSVAndTs
            else: 
                self.distBtwnSVAndVl = self.precomputeSquaredDistances(supportVectorInputVals, testData)
                cache = self.distBtwnSVAndVl

        correct: int = 0
        for index,(_, yTest) in enumerate(zip(testData, testLabel)):
            prediction: float = bValue
            for i in supportVectorIndexes:
                kVal = math.exp(cache[i][index]*denom)
                prediction += (alphas[i] * trLabel[i] * kVal)

            yPredict = 1 if prediction >= 0 else -1
            if yPredict == yTest: correct += 1

        return correct / len(testData)
    
    @staticmethod
    def createKey(X1: List[float], X2: List[float]) -> str:
        key = ""
        for x,y in zip(X1, X2):
            key += f"{x} {y} "
        return key

    @staticmethod
    def compareOverHyperParamsAndSigmas(trainData: List[List[float]], trainLabel: List[int], validationData: List[List[float]], validationLabel: List[int], \
                                        testData: List[List[float]], testLabel: List[int], \
                                            hyperList: List[float], sigmaList: List[float]) -> tuple[dict[float, dict[float, float]], dict[float, dict[float, float]]]:
        
        tempStructure: dict[float, dict[float, float]] = {x: {y: 0.0 for y in hyperList} for x in sigmaList}
        testAccuracyDict: dict[float, dict[float, float]] = copy.deepcopy(tempStructure)
        validAccuracyDict: dict[float, dict[float, float]] = copy.deepcopy(tempStructure)
        trainer: SVM_Slack_GaussianKernel = SVM_Slack_GaussianKernel(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
        for x in sigmaList:
            for y in hyperList:
                sol: dict[str, Any] = trainer.solveWithHyperParamAndSigma(y, x)
                testAccuracyDict[x][y] = trainer.computeKernelAccuracy(sol, x, True)
                validAccuracyDict[x][y] = trainer.computeKernelAccuracy(sol, x, False)
        return testAccuracyDict,validAccuracyDict
    
    
    
    @staticmethod
    def combinedTrainValidModel(trainData: List[List[float]], trainLabel: List[int], validationData: List[List[float]], validationLabel: List[int], \
                                testData: List[List[float]], testLabel: List[int], sigmaList: List[float], bestHyperForSigma: dict[float, float]) -> dict[float, float]:
        
        validDCopy: List[List[float]] = copy.deepcopy(validationData)
        validLCopy: List[int] = copy.deepcopy(validationLabel)
        trainData = trainData + validDCopy
        trainLabel = trainLabel + validLCopy
        trainer: SVM_Slack_GaussianKernel = SVM_Slack_GaussianKernel(trainData, trainLabel, [], [], testData, testLabel)
        accuracyForSigmaDict: dict[float, float] = {}
        for sig in sigmaList:
            hyperParam: float = bestHyperForSigma[sig]
            sol: dict[tuple, Any] = trainer.solveWithHyperParamAndSigma(hyperParam, sig)
            accuracyForSigmaDict[sig] = trainer.computeKernelAccuracy(sol, sig, True)
        return accuracyForSigmaDict
    
    @staticmethod
    def findBestCForASigma(validDict: dict[float, dict[float, float]]) -> dict[float, float]:
        sigmaToHyperDict: dict[float, float] = {}
        for keySig, acList in validDict.items():
            bestHyper: float = max(acList, key=acList.get)
            sigmaToHyperDict[keySig] = bestHyper
        return sigmaToHyperDict