from typing import List

import matplotlib.pyplot as plt

from SVM_Slack_With_AdaBoost import SVM_Slack_With_AdaBoost

def getInput(trainPath: str = "./heart_train.data", testPath: str = "./heart_test.data") -> \
        tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    print(f"Loading Training data from {trainPath}")
    trainData: List[List[float]] = []
    trainLabel: List[float] = []
    with open(trainPath, "r") as file:
        for line in file:
            valArr: List[float] = [1.0 if int(i) == 1 else -1.0 for i in "".join(line.split()).split(',')]
            trainData.append(valArr[1:])
            trainLabel.append(valArr[0])

    print(f"Loading Test data from {testPath}")
    testData: List[List[float]] = []
    testLabel: List[float] = []
    with open(testPath, "r") as file:
        for line in file:
            valArr: List[float] = [1.0 if int(i) == 1 else -1.0 for i in "".join(line.split()).split(',')]
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

    _, axs = plt.subplots(1,3, figsize=(15,5)) # type: ignore

    #Accuracy Plot
    axs[0].plot(iterationList, trainingAccuracy, color='blue', label='Training')
    axs[0].plot(iterationList, testAccuracy, color='red', label='Test')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_title('Accuracy over AdaBoost iterations')
    axs[0].legend()


    #Eps Plot
    axs[1].plot(iterationList, weightedError, color='red', label='Weighted Error')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Weighted Error')
    axs[1].set_title('Weighted Error over AdaBoost iterations')
    axs[1].legend()

    #Eps Plot
    axs[2].plot(iterationList, alphas, color='red', label='Alphas')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Alpha')
    axs[2].set_title('Alphas over AdaBoost iterations')
    axs[2].legend()

    # Display Plot
    plt.tight_layout() 
    plt.show() # type: ignore




if __name__ == "__main__":
    main()