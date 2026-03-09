from typing import Any, List

from SVM_Slack_Normal import SVM_Slack_Normal
from SVM_Slack_GaussianKernel import SVM_Slack_GaussianKernel
from SVM_Slack_Squared import SVM_Slack_Part1
from SVM_Slack_Max import SVM_Slack_Part2

hyperList: List[float] = [0.01,0.1,1.0,10.0,100.0,1000.0,10000.0,100000.0]
sigmaList: List[float] = [.001,.01,.1,1,10,100]

def computeAccuracy(solution: (dict[str, Any] | Any), data: List[List[float]], label: List[int]) -> float:
    wVector: List[float] = solution['x'][0:len(data[0])]
    bias: float = solution['x'][len(data[0])]
    
    classes: List[float] = [
        y * (sum([xi * wi for xi, wi in zip(x, wVector)]) + bias) 
        for x, y in zip(data, label)
    ]
    return sum([1 if x > 0 else 0 for x in classes]) / len(data)

def getInputData(path: str|None =None) -> tuple[List[List[float]], List[int]]:
    if path == None: path = "./magic.data"

    print(f"Loading from: {path}")

    inputData: List[List[float]] = []
    inputDataLabels: List[int] = []
    with open(path, "r") as file:
        for line in file:
            valArr: List[float] = [float(i) for i in "".join(line.split()).split(',')]
            inputData.append(valArr[:-1])
            inputDataLabels.append(int(valArr[-1]) if int(valArr[-1]) == 1 else -1) #Changes 0 classifications to -1
    
    #Assertion Tests (Can Ignore)
    testInput: List[float] = [90.4766,38.2659,3.6727,0.2042,0.1048,-47.7558,44.2203,23.7069,0.141,373.654]
    for index,x in enumerate(inputData[0]):
        assert x == testInput[index], "Failed to correctly get data"

    testOutput: List[int] = [-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1]
    for index,x in enumerate(inputDataLabels):
        assert x == testOutput[index], "Failed to correctly get data"
        if index == len(testOutput) -1: break

    return inputData, inputDataLabels

#Normal
def testNormal(trainData: List[List[float]], trainLabel: List[int], validationData: List[List[float]], validationLabel: List[int], testData: List[List[float]], testLabel: List[int]) -> str:
    testAc, validAc = SVM_Slack_Normal.compareOverHyperParams(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, hyperList, computeAccuracy
    )

    bestHyper: float = max(validAc, key=validAc.get)
    newModelAc = SVM_Slack_Normal.combinedTrainValidModel(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, bestHyper, computeAccuracy
    )

    return formatTestString("SVM with Slack Results", testAc, validAc, bestHyper, newModelAc)

#Gaussian
def testGaussian(trainData: List[List[float]], trainLabel: List[int], validationData: List[List[float]], validationLabel: List[int], testData: List[List[float]], testLabel: List[int]) -> str:
    testAc, validAc = SVM_Slack_GaussianKernel.compareOverHyperParamsAndSigmas(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, hyperList, sigmaList
    )

    bestHypersForSigma: dict[float, float] = SVM_Slack_GaussianKernel.findBestCForASigma(validAc)
    newModelAc = SVM_Slack_GaussianKernel.combinedTrainValidModel(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, sigmaList, bestHypersForSigma
    )

    return formatGaussString(testAc, validAc, bestHypersForSigma, newModelAc)    

#Part 1 
def testSlackSquared(trainData: List[List[float]], trainLabel: List[int], validationData: List[List[float]], validationLabel: List[int], testData: List[List[float]], testLabel: List[int]) -> str:
    testAc, validAc = SVM_Slack_Part1.compareOverHyperParams(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, hyperList, computeAccuracy
    )

    bestHyper = max(validAc, key=validAc.get)
    newModelAc = SVM_Slack_Part1.combinedTrainValidModel(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, bestHyper, computeAccuracy
    )

    return formatTestString("SVM with Slack Squared Results", testAc, validAc, bestHyper, newModelAc)

#Part 2 
def testSlackMax(trainData: List[List[float]], trainLabel: List[int], validationData: List[List[float]], validationLabel: List[int], testData: List[List[float]], testLabel: List[int]) -> str:
    testAc, validAc = SVM_Slack_Part2.compareOverHyperParams(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, hyperList, computeAccuracy
    )

    bestHyper = max(validAc, key=validAc.get)
    newModelAc = SVM_Slack_Part2.combinedTrainValidModel(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, bestHyper, computeAccuracy
    )

    return formatTestString("SVM with Max Slack Results", testAc, validAc, bestHyper, newModelAc)


def formatGaussString(testAc, validAc, bestHyper, newModelAc) -> str:
    resString: str = "Gaussian Kernel SVM Results\n"
    resString += f"Test Set Accuracy: (\"{{Sigma -> {{C -> Model Accuracy %}}}}\"): {testAc}\n" 
    resString += f"Validation Set Accuracy: (\"{{Sigma -> {{C -> Model Accuracy %}}}}\"): {validAc}\n"
    resString += f"Best Hyperparameter Per Sigma on Validation Set: {bestHyper}\n"
    resString += f"Combined Model Hyperparameter Per Sigma: {bestHyper}\n"
    resString += f"Combined Model Accuracy Per Sigma: {newModelAc}\n\n"
    return resString

def formatTestString(title: str, testAc: dict[float, float], validAc: dict[float, float], bestHyper: float, newModelAc: float) -> str:
    resString: str = title + "\n"
    resString += f"Test Set Accuracy: {testAc}\n" 
    resString += f"Validation Set Accuracy: {validAc}\n"
    resString += f"Best Hyperparameter on Validation Set: {bestHyper}\n" 
    resString += f"Combined Model Hyperparameter: {bestHyper}\n"
    resString += f"Combined Model Accuracy: {newModelAc}\n\n"
    return resString

def runTests(excludeList: List[int]|None=None, filePath: str|None=None, displayOutput: bool=True) -> str:
    data, labels = getInputData(filePath) 
    trainData, trainLabel = data[0:1800], labels[0:1800]
    validationData, validationLabel = data[1800:2400], labels[1800:2400]
    testData, testLabel = data[2400:], labels[2400:]

    resultString: str = "\n"

    if excludeList == None:
        excludeList = []

    for x in [0,1,2,3]: #0 is normal, 1 is gauss, 2 is squared, 3 is max
        if x in excludeList: continue

        match x:
            case 0: resultString += testNormal(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
            case 1: resultString += testGaussian(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
            case 2: resultString += testSlackSquared(trainData, trainLabel, validationData, validationLabel, testData, testLabel)
            case 3: resultString += testSlackMax(trainData, trainLabel, validationData, validationLabel, testData, testLabel)

    if displayOutput: print(resultString)
            
if __name__ == "__main__":
    runTests([0,2,3])