from typing import Any, List
from SVM_Slack_With_AdaBoost import SVM_Slack_With_AdaBoost
import matplotlib.pyplot as plt

def getInput(trainPath: str = "./heart_train.data", testPath: str = "./heart_test.data") -> \
        tuple[List[List[int]], List[int], List[List[int]], List[int]]:
    print(f"Loading Training data from {trainPath}")
    trainData: List[List[int]] = []
    trainLabel: List[int] = []
    with open(trainPath, "r") as file:
        for line in file:
            valArr: List[int] = [1 if int(i) == 1 else -1 for i in "".join(line.split()).split(',')]
            trainData.append(valArr[1:])
            trainLabel.append(valArr[0])

    print(f"Loading Test data from {testPath}")
    testData: List[List[int]] = []
    testLabel: List[int] = []
    with open(testPath, "r") as file:
        for line in file:
            valArr: List[int] = [1 if int(i) == 1 else -1 for i in "".join(line.split()).split(',')]
            testData.append(valArr[1:])
            testLabel.append(valArr[0])

    return (trainData, trainLabel, testData, testLabel)

def main():
    trData, trLabel, tsData, tsLabel = getInput()
    trainingAccuracy: List[float] = []
    testAccuracy: List[float] = []
    weightedError: List[float] = []
    alphas: List[float] = []
    iterationList: List[int] = list(range(1, 11))

    trainingResults = SVM_Slack_With_AdaBoost.runAdaBoostSVMAlgorithm(trData, trLabel)
    trainVotes = [0.0] * len(trData)
    testVotes = [0.0] * len(tsData)

    for x in range(len(trainingResults)):
        eps, alpha, w, b = trainingResults[x]

        for i in range(len(trData)):
            pred = 1 if (sum(xi * wi for xi, wi in zip(trData[i], w)) + b) >= 0 else -1
            trainVotes[i] += alpha * pred

        for i in range(len(tsData)):
            pred = 1 if (sum(xi * wi for xi, wi in zip(tsData[i], w)) + b) >= 0 else -1
            testVotes[i] += alpha * pred

        trAcc = sum(1 for i in range(len(trData)) if (1 if trainVotes[i] >= 0 else -1) == trLabel[i]) / len(trData)
        tsAcc = sum(1 for i in range(len(tsData)) if (1 if testVotes[i] >= 0 else -1) == tsLabel[i]) / len(tsData)


        trainingAccuracy.append(trAcc)
        testAccuracy.append(tsAcc)
        weightedError.append(eps)
        alphas.append(alpha)

        print(f"Iteration {x} epsilon {eps}")
        print(f"Iteration {x} alpha {alpha}")
        print(f"Iteration {x} weights {w}", end="")
        print(f"Iteration {x} bias {b}")
        print(f"Iteration {x} training accuracy {trAcc}")
        print(f"Iteration {x} testing accuracy {tsAcc}")
        print("\n")

    #Accuracy Plot
    plt.subplot(1, 3, 1)
    plt.plot(iterationList, trainingAccuracy, color='blue', label='Training')
    plt.plot(iterationList, testAccuracy, color='red', label='Test')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over AdaBoost iterations')
    plt.legend()

    #Eps Plot
    plt.subplot(1, 3, 2)
    plt.plot(iterationList, weightedError, color='red', label='Weighted Error')
    plt.xlabel('Iteration')
    plt.ylabel('Weighted Error')
    plt.title('Weighted Error over AdaBoost iterations')
    plt.legend()

    #Eps Plot
    plt.subplot(1, 3, 3)
    plt.plot(iterationList, alphas, color='red', label='Alphas')
    plt.xlabel('Iteration')
    plt.ylabel('Alpha')
    plt.title('Alphas over AdaBoost iterations')
    plt.legend()

    # Display Plot
    plt.show()




if __name__ == "__main__":
    main()