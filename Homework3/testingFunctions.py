import copy
import random
from typing import List

from queryDataset import QuerySegment, AttributeValueWrapper, queryDataUsingQuerySegments, prettyPrintRes
from descisionTree import DescisionTree

def executeQueryTest(trainData: List[tuple[str, List[str]]]) -> None:
    #These are all the unused branches except the last one 
    #Each segment has a list of AttributeValueWrapper's
    #Each wrapper corresponds to a specific attribute in the dataset you want to query for 
    #Ex) AttributeValueWrapper(4, 'n') means that only values with Odor equal to 'n' will be selected for that segment
    query: List[QuerySegment] = [
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'u')
        ]),
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'm')
        ]),
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'u')
        ]),
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'l'),
            AttributeValueWrapper(20, 'a'),
        ]),
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'l'),
            AttributeValueWrapper(20, 'n'),
        ]),
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'l'),
            AttributeValueWrapper(20, 's'),
        ]),
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'l'),
            AttributeValueWrapper(20, 'y'),
        ]),
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'd'),
            AttributeValueWrapper(20, 'a'),
        ]),
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'd'),
            AttributeValueWrapper(20, 'c'),
        ]),
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'd'),
            AttributeValueWrapper(20, 'n'),
        ]),
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'd'),
            AttributeValueWrapper(20, 's'),
        ]),
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'd'),
            AttributeValueWrapper(20, 'y'),
        ]),
    ]

    res: List[List[str]] = queryDataUsingQuerySegments(query, trainData)
    print(len(res))
    prettyPrintRes(res)

    query: List[QuerySegment] = [
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'l'),
        ]),
        QuerySegment([
            AttributeValueWrapper(4, 'n'),
            AttributeValueWrapper(19, 'w'),
            AttributeValueWrapper(21, 'd'),
        ]),
    ]

    res: List[tuple[int, List[str]]] = queryDataUsingQuerySegments(query, trainData)
    print(len(res))
    prettyPrintRes(res)

def executeReshuffleTest(trainData: List[tuple[str, List[str]]], testData: List[tuple[str, List[str]]]) -> None:
    lenOriginalTrain: int = len(trainData)
    copiedCombinedData: List[tuple[str, List[str]]] = copy.deepcopy(trainData) + copy.deepcopy(testData)
    random.shuffle(copiedCombinedData)

    trainDataNew = copiedCombinedData[0:lenOriginalTrain]
    testDataNew = copiedCombinedData[lenOriginalTrain:]

    DescisionTree.trainDTree(trainDataNew, testDataNew)

def executeNReshuffleTests(trainData: List[tuple[str, List[str]]], testData: List[tuple[str, List[str]]], reshuffles: int=100) -> None:
    lenOriginalTrain: int = len(trainData)
    copiedCombinedData: List[tuple[str, List[str]]] = copy.deepcopy(trainData) + copy.deepcopy(testData)
    trCumulativeAc = 0
    tsCumulativeAc = 0
    for z in range(reshuffles):
        random.shuffle(copiedCombinedData)
        trainDataNew = copiedCombinedData[0:lenOriginalTrain]
        testDataNew = copiedCombinedData[lenOriginalTrain:]
        x,y = DescisionTree.trainDTree(trainDataNew, testDataNew, False)
        trCumulativeAc += x
        tsCumulativeAc += y

        if y < 1:
            print(f"Testing Accuracy for iteration {z+1} was {y}")

    print(f"After {reshuffles} shuffles the average training accuracy was {trCumulativeAc/reshuffles} and average testing accuracy was {tsCumulativeAc/reshuffles}")
        