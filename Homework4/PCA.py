import numpy as np
import numpy.typing as npt
from typing import List

from SVM_Slack_GaussianKernel import SVM_Slack_GaussianKernel

TOP_K_EVALUES_LIST = [.99,.95,.9,.80,.75]

hyperList: List[float] = [0.01,0.1,1.0,10.0,100.0,1000.0,10000.0,100000.0]
sigmaList: List[float] = [.001,.01,.1,1,10,100]

class PCA:
    def __init__(self, trainingData: npt.NDArray[np.float64], featureCount: int) -> None:
        self.trData = trainingData
        self.features = featureCount
        self.numDataPoints = len(self.trData)

        self.kSizes = []
        self.mean = 0.0
        self.W = None
        self.components = None
        self.PerformPCA()


        assert self.numDataPoints == 6000, "Mismatch. Fix this."

    def PerformPCA(self):
        self.mean = np.mean(self.trData, axis=0, keepdims=True)
        self.W = self.trData - self.mean
        eVectors, eValues, vh = np.linalg.svd(self.W, full_matrices=False)
        self.components = vh
        eValues = ((eValues**2) / (self.W.shape[0]))
        totalVariance = np.sum(eValues)
        cumulativeVar = np.cumsum(eValues) / totalVariance

        print(f"Top 6 Eigenvalues: {eValues[:6]}")

        kValues = {}
        for x in TOP_K_EVALUES_LIST:
            k = np.searchsorted(cumulativeVar, x) + 1
            kValues[int(x*100)] = k

        for key,value in kValues.items(): 
            print(f"Top {value} eigenvalues explain {key}% of variance")
            self.kSizes.append(value)

    @staticmethod
    def formatGaussString(testAc, validAc, bestHyper, newModelAc, value) -> str:
        resString: str = f"Gaussian Kernel SVM Results for K={value}\n"
        resString += f"Test Set Accuracy: (\"{{Sigma -> {{C -> Model Accuracy %}}}}\"): {testAc}\n" 
        resString += f"Validation Set Accuracy: (\"{{Sigma -> {{C -> Model Accuracy %}}}}\"): {validAc}\n"
        resString += f"Best Hyperparameter Per Sigma on Validation Set: {bestHyper}\n"
        resString += f"Combined Model Accuracy Per Sigma: {newModelAc}\n\n"
        return resString

    def trainSVMs(self, trLabel, tsData, tsLabel, vdData, vdLabel):
        for value in self.kSizes:
            vk = self.components[:value, :].T

            trDataP = self.W @ vk
            vdDataP = (vdData - self.mean) @ vk
            tsDataP = (tsData - self.mean) @ vk
            
            testAc, validAc = SVM_Slack_GaussianKernel.compareOverHyperParamsAndSigmas(trDataP, trLabel, \
                    vdDataP, vdLabel, tsDataP, tsLabel, hyperList, sigmaList)
            bestHypersForSigma: dict[float, float] = SVM_Slack_GaussianKernel.findBestCForASigma(validAc)
            newModelAc = SVM_Slack_GaussianKernel.combinedTrainValidModel(
                trDataP, trLabel, vdDataP, vdLabel, tsDataP, tsLabel, sigmaList, bestHypersForSigma
            )
            print(PCA.formatGaussString(testAc, validAc, bestHypersForSigma, newModelAc, value))