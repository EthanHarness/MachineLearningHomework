#Code written by Ethan Harness
#This file is the main runner for the code
#descisionTree.py has main descision tree functionality
#queryDataset.py has functions to help test accuracy of descisionTree.py and just creates a querying system to find values
#testingFunctions.py has 2 main tests we can optionally run (Can customize to test behavior)
#First function queries input and finds specific values
#Second has a function to combine train and test data then train a new model by shuffling order

from typing import List

from descisionTree import Attribute, DescisionTree
from testingFunctions import executeQueryTest, executeReshuffleTest

#Helper to make it easier to reason about assumptions made in code
def makeAssertionOnInput(featureList: List[str]) -> None:
    for index, x in enumerate(featureList):
        assert x in Attribute.getDomainOfFeature(index), f"Feature List, {featureList}, has domain issue for feature {index}"
    return

def getInputData(testPath: str="", trainPath: str="") -> tuple[List[tuple[str, List[str]]], List[tuple[str, List[str]]]]:
    if testPath == "": testPath = "./mush_test.data"
    if trainPath == "": trainPath = "./mush_train.data"

    print(f"Loading train from: {trainPath}")
    print(f"Loading test from: {testPath}")

    trainData: List[tuple[str, List[str]]] = []
    testData: List[tuple[str, List[str]]] = []
    with open(trainPath, "r") as file:
        for line in file:
            line = line.strip()
            dataArr: List[str] = line.split(",")
            trainData.append((dataArr[0], dataArr[1:]))
            makeAssertionOnInput(dataArr[1:])

    with open(testPath, "r") as file:
        for line in file:
            line = line.strip()
            dataArr: List[str] = line.split(",")
            testData.append((dataArr[0], dataArr[1:]))
            makeAssertionOnInput(dataArr[1:])

    return (trainData, testData)

def main(skipTests: List[int] = []):
    trainData, testData = getInputData()
    print("Descision Tree using original dataset orderings....")
    
    #Main function that trains the default descision tree
    #Can run other tests and modify them to test behavior
    DescisionTree.trainDTree(trainData, testData) 

    #Just a test that queries input data and returns resutls
    if 0 not in skipTests: executeQueryTest(trainData) 

    #Just a test that combines both train and test, reshuffles, then retrains and reports accuracy 
    if 1 not in skipTests: executeReshuffleTest(trainData, testData) 

if __name__ == "__main__":
    main([0, 1])