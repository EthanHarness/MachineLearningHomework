import numpy as np
import numpy.typing as npt
from typing import List

from PCA import PCA

TRAIN_LINE_COUNT_CONSTANT = 6000
TEST_LINE_COUNT_CONSTANT = 500
VALIDATION_LINE_COUNT_CONSTANT = 500
FEATURE_COUNT_CONSTANT = 5000


def getInput(trainPath: str = "./gisette_train.data", testPath: str = "./gisette_test.data", validPath: str = "./gisette_valid.data") \
    -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    
    print(f"Loading Training data from {trainPath}")
    print(f"Loading Test data from {testPath}")
    print(f"Loading Validation data from {validPath}")

    trData: npt.NDArray[np.int32] = np.empty((TRAIN_LINE_COUNT_CONSTANT, FEATURE_COUNT_CONSTANT), dtype=np.int32)
    trLabels: npt.NDArray[np.int32] = np.empty((TRAIN_LINE_COUNT_CONSTANT), dtype=np.int32)
    with open(trainPath, "r") as file:
        for index,line in enumerate(file):
            lData: List[str] = "".join(line.split()).split(',')
            trData[index] = np.array([np.int32(i) for i in lData[1:]], dtype=np.int32)
            trLabels[index] = int(lData[0])

    tsData: npt.NDArray[np.int32] = np.empty((TEST_LINE_COUNT_CONSTANT, FEATURE_COUNT_CONSTANT), dtype=np.int32)
    tsLabels: npt.NDArray[np.int32] = np.empty((TEST_LINE_COUNT_CONSTANT), dtype=np.int32)
    with open(testPath, "r") as file:
        for index,line in enumerate(file):
            lData: List[str] = "".join(line.split()).split(',')
            tsData[index] = np.array([np.int32(i) for i in lData[1:]], dtype=np.int32)
            tsLabels[index] = int(lData[0])

    vdData: npt.NDArray[np.int32] = np.empty((VALIDATION_LINE_COUNT_CONSTANT, FEATURE_COUNT_CONSTANT), dtype=np.int32)
    vdLabels: npt.NDArray[np.int32] = np.empty((VALIDATION_LINE_COUNT_CONSTANT), dtype=np.int32)
    with open(validPath, "r") as file:
        for index,line in enumerate(file):
            lData: List[str] = "".join(line.split()).split(',')
            vdData[index] = np.array([np.int32(i) for i in lData[1:]], dtype=np.int32)
            vdLabels[index] = int(lData[0])

    return (trData, trLabels, tsData, tsLabels, vdData, vdLabels) 

def scaleData(trainingData: npt.NDArray[np.int32]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    column_means: npt.NDArray[np.float64] = np.mean(trainingData, axis=0)
    col_deviations: npt.NDArray[np.float64] = np.std(trainingData, axis=0)
    safe_deviations: npt.NDArray[np.float64] = np.where(col_deviations == 0, 1, col_deviations)

    scaled = (trainingData - column_means) / safe_deviations
    return scaled, column_means, safe_deviations



def main():
    trData, trLabel, tsData, tsLabel, vdData, vdLabel = getInput()
    scaledTrData, mean, std = scaleData(trData)

    scaledTsData = (tsData - mean) / std
    scaledVdData = (vdData - mean) / std

    pca = PCA(scaledTrData, FEATURE_COUNT_CONSTANT)
    pca.trainSVMs(trLabel, scaledTsData, tsLabel, scaledVdData, vdLabel)

if __name__ == "__main__":
    main()