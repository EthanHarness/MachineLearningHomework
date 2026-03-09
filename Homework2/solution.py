from SVM_Slack_Normal import SVM_Slack_Normal
from SVM_Slack_GaussianKernel import SVM_Slack_GaussianKernel
from SVM_Slack_Squared import SVM_Slack_Part1
from SVM_Slack_Max import SVM_Slack_Part2

# hyperList = [0.01,0.1,1.0,10.0,100.0,1000.0,10000.0,100000.0]
# sigmaList = [.001,.01,.1,1,10,100]
hyperList = [0.01,0.1,1.0,]
sigmaList = [.001,.01,.1]

def computeAccuracy(solution, data, label):
    wVector = solution['x'][0:len(data[0])]
    bias = solution['x'][len(data[0])]
    
    classes = [
        y * (sum([xi * wi for xi, wi in zip(x, wVector)]) + bias) 
        for x, y in zip(data, label)
    ]
    return sum([1 if x > 0 else 0 for x in classes]) / len(data)

def getInputData(path=None):
    if path == None: path = "./magic.data"

    print(f"Loading from: {path}")

    inputData = []
    inputDataLabels = []
    with open(path, "r") as file:
        for line in file:
            valArr = "".join(line.split())
            valArr = [float(i) for i in valArr.split(',')]
            valArr[-1] = int(valArr[-1]) if int(valArr[-1]) == 1 else -1 #Changes 0 classifications to -1
            inputData.append(valArr[:-1])
            inputDataLabels.append(valArr[-1])
            
    return (inputData, inputDataLabels)

#Normal
def testNormal(trainData, trainLabel, validationData, validationLabel, testData, testLabel):
    testAc, validAc = SVM_Slack_Normal.compareOverHyperParams(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, hyperList, computeAccuracy
    )

    bestHyper = max(validAc, key=validAc.get)
    newModelAc = SVM_Slack_Normal.combinedTrainValidModel(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, bestHyper, computeAccuracy
    )

    return formatTestString("SVM with Slack Results", False, testAc, validAc, bestHyper, newModelAc)

#Gaussian
def testGaussian(trainData, trainLabel, validationData, validationLabel, testData, testLabel):
    testAc, validAc = SVM_Slack_GaussianKernel.compareOverHyperParamsAndSigmas(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, hyperList, sigmaList
    )

    bestHypersForSigma = SVM_Slack_GaussianKernel.findBestCForASigma(validAc)
    newModelAc = SVM_Slack_GaussianKernel.combinedTrainValidModel(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, sigmaList, bestHypersForSigma
    )

    return formatTestString("Gaussian Kernel SVM Results", True, testAc, validAc, bestHypersForSigma, newModelAc)    

#Part 1 
def testSlackSquared(trainData, trainLabel, validationData, validationLabel, testData, testLabel):
    testAc, validAc = SVM_Slack_Part1.compareOverHyperParams(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, hyperList, computeAccuracy
    )

    bestHyper = max(validAc, key=validAc.get)
    newModelAc = SVM_Slack_Part1.combinedTrainValidModel(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, bestHyper, computeAccuracy
    )

    return formatTestString("SVM with Slack Squared Results", False, testAc, validAc, bestHyper, newModelAc)

#Part 2 
def testSlackMax(trainData, trainLabel, validationData, validationLabel, testData, testLabel):
    testAc, validAc = SVM_Slack_Part2.compareOverHyperParams(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, hyperList, computeAccuracy
    )

    bestHyper = max(validAc, key=validAc.get)
    newModelAc = SVM_Slack_Part2.combinedTrainValidModel(
        trainData, trainLabel, validationData, validationLabel, testData, testLabel, bestHyper, computeAccuracy
    )

    return formatTestString("SVM with Max Slack Results", False, testAc, validAc, bestHyper, newModelAc)


def formatTestString(title, isGauss, testAc, validAc, bestHyper, newModelAc):
    resString = title
    resString += f"\nTest Set Accuracy: {testAc}\n"
    resString += f"Validation Set Accuracy: {validAc}\n"
    resString += f"Best Hyperparameter on Validation Set: {bestHyper}\n" if not isGauss else f"Best Hyperparameter Per Sigma on Validation Set: {bestHyper}\n"
    resString += f"Combined Model Hyperparameter: {bestHyper}\n" if not isGauss else f"Combined Model Hyperparameter Per Sigma: {bestHyper}\n"
    resString += f"Combined Model Accuracy: {newModelAc}\n\n" if not isGauss else f"nCombined Model Accuracy Per Sigma: {newModelAc}\n\n"
    return resString

def runTests(excludeList=None, filePath=None, displayOutput=True):
    data, labels = getInputData(filePath)
    trainData, trainLabel = data[0:1800], labels[0:1800]
    validationData, validationLabel = data[1800:2400], labels[1800:2400]
    testData, testLabel = data[2400:], labels[2400:]

    resultString = "\n"

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
    runTests()