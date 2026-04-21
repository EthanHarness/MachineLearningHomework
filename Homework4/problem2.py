import numpy as np
import numpy.typing as npt
from typing import List

from KMeans import KMeans
from KMeansPlusPlus import KMeansPlusPlus

LINE_COUNT_CONSTANT = 340
FEATURE_COUNT_CONSTANT = 14

ITERATIONS = 200

CLASS_COUNT = 36

K_LIST = [20,25,30,35,40]

def getInput(trainPath: str = "./leaf.data") -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    print(f"Loading Training data from {trainPath}")

    trData: npt.NDArray[np.float64] = np.empty((LINE_COUNT_CONSTANT, FEATURE_COUNT_CONSTANT), dtype=np.float64)
    trLabels: npt.NDArray[np.int32] = np.empty((LINE_COUNT_CONSTANT), dtype=np.int32)

    with open(trainPath, "r") as file:
        for index,line in enumerate(file):
            lData: List[str] = "".join(line.split()).split(',')
            trData[index] = np.array([np.float64(i) for i in lData[1:]], dtype=np.float64)
            trLabels[index] = int(lData[0])

    return (trData, trLabels)

def scaleData(trainingData: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    column_means: npt.NDArray[np.float64] = np.mean(trainingData, axis=0)
    col_deviations: npt.NDArray[np.float64] = np.std(trainingData, axis=0)

    def scaleRow(row: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array([(np.float64(i - column_means[index])/col_deviations[index]) for index,i in enumerate(row)], dtype=np.float64)
    
    res: npt.NDArray[np.float64] = np.apply_along_axis(scaleRow, axis=1, arr=trainingData)

    return res

def averageLabels(clusterLabelToDataPointLabel: dict[int, List[int]]) -> float:
    clusters: int = len(clusterLabelToDataPointLabel)
    cumulativeSum: int = 0
    for _,value in clusterLabelToDataPointLabel.items():
        cumulativeSum += max(value)
    return cumulativeSum / clusters

def runForRegularKMeans(trData: npt.NDArray[np.float64], labels: npt.NDArray[np.int32]):
    print("Regular K Means Algorithm Results")
    for k in K_LIST:
        model = KMeans(trData, ITERATIONS, k)
        results: npt.NDArray[np.float64] = model.getTrainingResults()
        mean: np.float64 = results.mean()
        std: np.float64 = results.std()
        
        def createCountArray() -> List[int]:
            return [0]*CLASS_COUNT
        clusterLabelToDataPointLabel: dict[int, List[int]] = {x: createCountArray() for x in range(k)}
        distances = np.linalg.norm(trData[:, np.newaxis] - model.clusterCenters, axis=2)
        clusterLabels = np.argmin(distances, axis=1)

        assert len(clusterLabels) == len(labels), "Fix length mismatch"
        for clusterLabel, dataLabel in zip(clusterLabels, labels):
            clusterLabelToDataPointLabel[clusterLabel][dataLabel-1] += 1

        print(f"k={k} mean={mean} std={std}")
        print(f"Average maximum labels for clusters is: {averageLabels(clusterLabelToDataPointLabel)}")

def runForKMeansPlusPlus(trData: npt.NDArray[np.float64], labels: npt.NDArray[np.int32]):
    print("K Means Plus Plus Algorithm Results")
    for k in K_LIST:
        model = KMeansPlusPlus(trData, ITERATIONS, k)
        results: npt.NDArray[np.float64] = model.getTrainingResults()
        mean: np.float64 = results.mean()
        std: np.float64 = results.std()

        def createCountArray() -> List[int]:
            return [0]*CLASS_COUNT
        clusterLabelToDataPointLabel = {x: createCountArray() for x in range(k)}
        distances = np.linalg.norm(trData[:, np.newaxis] - model.clusterCenters, axis=2)
        clusterLabels = np.argmin(distances, axis=1)

        assert len(clusterLabels) == len(labels), "Fix length mismatch"
        for clusterLabel, dataLabel in zip(clusterLabels, labels):
            clusterLabelToDataPointLabel[clusterLabel][dataLabel-1] += 1

        print(f"k={k} mean={mean} std={std}")
        print(f"Average maximum labels for clusters is: {averageLabels(clusterLabelToDataPointLabel)}")



def main():
    trData, trLabels = getInput()
    scaledTrData: npt.NDArray[np.float64] = scaleData(trData)

    runForRegularKMeans(scaledTrData, trLabels)
    print()
    runForKMeansPlusPlus(scaledTrData, trLabels)

if __name__ == "__main__":
    main()