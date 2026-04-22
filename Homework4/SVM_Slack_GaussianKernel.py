import numpy as np
from typing import Any, Dict, Tuple
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False # type: ignore

class SVM_Slack_GaussianKernel:
    def __init__(self, trData: np.ndarray, trLabel: np.ndarray, 
                 vData: np.ndarray, vLabel: np.ndarray, 
                 tsData: np.ndarray, tsLabel: np.ndarray) -> None:
        
        self.trData = np.array(trData)
        self.trLabel = np.array(trLabel).astype(float)
        self.vData = np.array(vData)
        self.vLabel = np.array(vLabel).astype(float)
        self.tsData = np.array(tsData)
        self.tsLabel = np.array(tsLabel).astype(float)
        
        assert self.trData.shape[0] == 6000, "Input Data wrong size"

        # Precompute squared distances for training data
        self.distBtwnTrData = self.precomputeSquaredDistances(self.trData, self.trData)
        
        # Caches for distance matrices
        self.distBtwnSVAndTs: np.ndarray = np.array([])
        self.distBtwnSVAndVl: np.ndarray = np.array([])

    def precomputeSquaredDistances(self, setA: np.ndarray, setB: np.ndarray) -> np.ndarray:
        sq_norms_A = np.sum(setA**2, axis=1).reshape(-1, 1)
        sq_norms_B = np.sum(setB**2, axis=1).reshape(1, -1)
        dist_mat = sq_norms_A + sq_norms_B - 2 * np.dot(setA, setB.T)
        return dist_mat * -1.0

    def constructPMatrix(self, sigma: float) -> np.ndarray:
        matSize = self.trData.shape[0]
        denom = 1.0 / (2.0 * sigma**2)
        
        kernel_mat = np.exp(self.distBtwnTrData * denom)
        
        P = np.outer(self.trLabel, self.trLabel) * kernel_mat
        return P
    
    def constructQMatrix(self) -> np.ndarray:
        return np.full((self.trData.shape[0],), -1.0)
    
    def constructHMatrix(self, hyperParam: float) -> np.ndarray:
        matSize = self.trData.shape[0]
        return np.hstack([np.full(matSize, hyperParam), np.zeros(matSize)])
    
    def constructGMatrix(self) -> np.ndarray:
        matSize = self.trData.shape[0]
        return np.vstack([np.eye(matSize), -np.eye(matSize)])

    def constructAMatrix(self) -> np.ndarray:
        return self.trLabel.reshape(1, -1)
    
    def solveWithHyperParamAndSigma(self, hyperParam: float, sigma: float) -> Dict[str, Any]:
        P = matrix(self.constructPMatrix(sigma), tc='d')
        q = matrix(self.constructQMatrix(), tc='d')
        G = matrix(self.constructGMatrix(), tc='d')
        h = matrix(self.constructHMatrix(hyperParam), tc='d')
        A = matrix(self.constructAMatrix(), tc='d')
        b = matrix(0.0, tc='d')
        
        return solvers.qp(P, q, G, h, A, b)
    
    def computeKernelAccuracy(self, sol: Dict[str, Any], sigma: float, mode: bool) -> float:
        alphas = np.array(sol['x']).flatten()
        denom = 1.0 / (2.0 * sigma**2)
        
        testData = self.tsData if mode else self.vData
        testLabel = self.tsLabel if mode else self.vLabel
        
        sv_mask = alphas > 1e-5
        sv_alphas = alphas[sv_mask]
        sv_labels = self.trLabel[sv_mask]
        sv_inputs = self.trData[sv_mask]
        
        dist_sv_sv = self.distBtwnTrData[sv_mask][:, sv_mask]
        kernel_sv_sv = np.exp(dist_sv_sv * denom)
        
        kernel_sum_sv = np.dot(kernel_sv_sv, (sv_alphas * sv_labels))
        bValue = np.mean(sv_labels - kernel_sum_sv)

        dist_sv_test = self.precomputeSquaredDistances(sv_inputs, testData)
        kernel_sv_test = np.exp(dist_sv_test * denom)

        predictions = np.dot(kernel_sv_test.T, (sv_alphas * sv_labels)) + bValue
        yPredict = np.where(predictions >= 0, 1, -1)

        return float(np.mean(yPredict == testLabel))

    @staticmethod
    def compareOverHyperParamsAndSigmas(trainData, trainLabel, validationData, validationLabel, 
                                        testData, testLabel, hyperList, sigmaList):
        
        testAccuracyDict = {s: {h: 0.0 for h in hyperList} for s in sigmaList}
        validAccuracyDict = {s: {h: 0.0 for h in hyperList} for s in sigmaList}
        
        trainer = SVM_Slack_GaussianKernel(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
        
        for sig in sigmaList:
            for hyp in hyperList:
                sol = trainer.solveWithHyperParamAndSigma(hyp, sig)
                testAccuracyDict[sig][hyp] = trainer.computeKernelAccuracy(sol, sig, True)
                validAccuracyDict[sig][hyp] = trainer.computeKernelAccuracy(sol, sig, False)
                
        return testAccuracyDict, validAccuracyDict

    @staticmethod
    def combinedTrainValidModel(trainData, trainLabel, validationData, validationLabel, 
                                testData, testLabel, sigmaList, bestHyperForSigma):
        
        combinedData = np.vstack([trainData, validationData])
        combinedLabel = np.hstack([trainLabel, validationLabel])
        
        trainer = SVM_Slack_GaussianKernel(combinedData[:6000], combinedLabel[:6000], [], [], testData, testLabel)
        
        accuracyForSigmaDict = {}
        for sig in sigmaList:
            hyp = bestHyperForSigma[sig]
            sol = trainer.solveWithHyperParamAndSigma(hyp, sig)
            accuracyForSigmaDict[sig] = trainer.computeKernelAccuracy(sol, sig, True)
        return accuracyForSigmaDict
    
    @staticmethod
    def findBestCForASigma(validDict: Dict[float, Dict[float, float]]) -> Dict[float, float]:
        sigmaToHyperDict: Dict[float, float] = {}
        
        for sigma, hyperResults in validDict.items():
            bestHyper = max(hyperResults, key=hyperResults.get)
            sigmaToHyperDict[sigma] = bestHyper
            
        return sigmaToHyperDict