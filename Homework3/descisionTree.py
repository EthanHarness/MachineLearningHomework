from __future__ import annotations
import random
import copy
from typing import List
import math

CAP_SHAPE_DOMAIN: List[str] = ['b','c','x','f','k','s']
CAP_SURFACE_DOMAIN: List[str] = ['f','g','y','s']
CAP_COLOR_DOMAIN: List[str] = ['n','b','c','g','r','p','u','e','w','y']
BRUISES_DOMAIN: List[str] = ['t','f']
ODOR_DOMAIN: List[str] = ['a','l','c','y','f','m','n','p','s']
GILL_ATTCH_DOMAIN: List[str] = ['a','f','d','n']
GILL_SPACE_DOMAIN: List[str] = ['c','w','d']
GILL_SIZE_DOMAIN: List[str] = ['b','n']
GILL_COLOR_DOMAIN: List[str] = ['k','n','b','h','g','r','o','p','u','e','w','y']
STALK_SHAPE_DOMAIN: List[str] = ['e','t']
STALK_ROOT_DOMAIN: List[str] = ['b','c','u','e','z','r', 'm'] 
STALK_SURFACE_ABOVE_DOMAIN: List[str] = ['f','y','k','s']
STALK_SURFACE_BELOW_DOMAIN: List[str] = ['f','y','k','s']
STALK_COLOR_ABOVE_DOMAIN: List[str] = ['n','b','c','g','o','p','e','w','y']
STALK_COLOR_BELOW_DOMAIN: List[str] = ['n','b','c','g','o','p','e','w','y']
VEIL_TYPE_DOMAIN: List[str] = ['p','u']
VEIL_COLOR_DOMAIN: List[str] = ['n','o','w','y']
RING_NUMBER_DOMAIN: List[str] = ['n','o','t']
RING_TYPE_DOMAIN: List[str] = ['c','e','f','l','n','p','s','z',]
SPORE_PRIINT_DOMAIN: List[str] = ['k','n','b','h','r','o','u','w','y']
POPULATION_DOMAIN: List[str] = ['a','c','n','s','v','y']
HABITAT_DOMAIN: List[str] = ['g','l','m','p','u','w','d']

LABEL_DOMAIN: List[str] = ['p','e']
NUMBER_OF_FEATURES: int = 22
LOG_BASE: int = 10


class Attribute:
    def __init__(self, featureNumber: int, label: str = "Top") -> None:
        self.attributeToSplitOn: int = featureNumber
        self.label: str = label
        self.childrenAttributes: List[Attribute] = []

    @staticmethod
    def getDomainOfFeature(featureNumber: int) -> List[str]:
        match featureNumber:
            case 0: return CAP_SHAPE_DOMAIN
            case 1: return CAP_SURFACE_DOMAIN
            case 2: return CAP_COLOR_DOMAIN
            case 3: return BRUISES_DOMAIN
            case 4: return ODOR_DOMAIN
            case 5: return GILL_ATTCH_DOMAIN
            case 6: return GILL_SPACE_DOMAIN
            case 7: return GILL_SIZE_DOMAIN
            case 8: return GILL_COLOR_DOMAIN
            case 9: return STALK_SHAPE_DOMAIN
            case 10: return STALK_ROOT_DOMAIN
            case 11: return STALK_SURFACE_ABOVE_DOMAIN
            case 12: return STALK_SURFACE_BELOW_DOMAIN
            case 13: return STALK_COLOR_ABOVE_DOMAIN
            case 14: return STALK_COLOR_BELOW_DOMAIN
            case 15: return VEIL_TYPE_DOMAIN
            case 16: return VEIL_COLOR_DOMAIN
            case 17: return RING_NUMBER_DOMAIN
            case 18: return RING_TYPE_DOMAIN
            case 19: return SPORE_PRIINT_DOMAIN
            case 20: return POPULATION_DOMAIN
            case 21: return HABITAT_DOMAIN

    def addAttributeSplitToChildren(self, attr: Attribute) -> None:
        self.childrenAttributes.append(attr)

    def printAttributeTree(self, numTabs: int = 0) -> str:
        printVal: str = str(self.attributeToSplitOn)
        if self.attributeToSplitOn == -1: printVal = 'p'
        elif self.attributeToSplitOn == -2: printVal = 'e'

        tabStringP: str = "\t" * numTabs
        resString: str = tabStringP + self.label + " " + printVal + "\n"
        for x in self.childrenAttributes:
            resString += x.printAttributeTree(numTabs+1)
        return resString
    
    def addClassLabel(self, label: str) -> None:
        self.label = label

class DescisionTree:
    @staticmethod
    def createSplits(label: List[str], featureList: List[List[str]]) -> Attribute:
        if len(label) == 1 or DescisionTree.labelsHave1Val(label): return Attribute(-1 if label[0] == 'p' else -2, label[0])

        splitAttr: int = DescisionTree.findAttributeToSplitOn(label, featureList)
        splitDomain: List[str] = Attribute.getDomainOfFeature(splitAttr)
        attr: Attribute = Attribute(splitAttr)

        divisions: dict[str, tuple[List[str], List[List[str]]]] = {x:([], []) for x in splitDomain}
        for l, fList in zip(label, featureList):
            divisions[fList[splitAttr]][0].append(l) 
            divisions[fList[splitAttr]][1].append(fList)
        
        for key in divisions.keys():
            if len(divisions[key][0]) == 0: continue #Skips over any unused keys in domain of feature
            attrChild = DescisionTree.createSplits(divisions[key][0], divisions[key][1])
            attr.addAttributeSplitToChildren(attrChild)
            attrChild.addClassLabel(key)

        return attr
            

    @staticmethod
    def entropyPoisonous(labels: List[str]) -> float:
        poisonSamples: int = len([x for x in labels if x == 'p'])
        samples: int = len(labels)
        if (poisonSamples == 0 or poisonSamples == samples): return 0

        p: float = poisonSamples / samples
        q: float = 1 - p

        return -1*((p*math.log(p,LOG_BASE)) + (q*math.log(q,LOG_BASE)))
    
    @staticmethod
    def entropyGivenAttribute(labels: List[str], featureList: List[List[str]], featureNumber: int) -> float:
        domain: List[str] = Attribute.getDomainOfFeature(featureNumber)
        entropy: float = 0.0
        samples: int = len(labels)

        for attribute in domain:
            attrCount: int = len([1 for x in featureList if x[featureNumber] == attribute])

            if attrCount == 0: continue

            pXeqlx: float = attrCount / samples
            cummulativeSum: float = 0
            for label in LABEL_DOMAIN:
                condCount: int = len([1 for labelVal,features in zip(labels, featureList) if labelVal == label and features[featureNumber] == attribute])

                if condCount == 0: continue

                pYGivenX: float = condCount/attrCount
                logpYGivenX: float = math.log(pYGivenX, LOG_BASE)
                cummulativeSum += (pYGivenX*logpYGivenX)
            entropy += cummulativeSum*pXeqlx

        return entropy*-1
    
    @staticmethod
    def findInformationGainForAttribute(labels: List[str], featureList: List[List[str]], featureNumber: int) -> float:
        condEntropy: float = DescisionTree.entropyGivenAttribute(labels, featureList, featureNumber)
        entropyPoisonous: float = DescisionTree.entropyPoisonous(labels)
        return entropyPoisonous - condEntropy
    
    @staticmethod
    def findAttributeToSplitOn(labels: List[str], featureList: List[List[str]]) -> int:
        maxInformationGain: float = 0
        featureToSplitOn: int = 0
        for x in range(NUMBER_OF_FEATURES):
            ig: float = DescisionTree.findInformationGainForAttribute(labels, featureList, x)
            if ig >= maxInformationGain:
                maxInformationGain = ig
                featureToSplitOn = x

        return featureToSplitOn
    
    @staticmethod
    def labelsHave1Val(labels: List[str]):
        return len(set(labels)) == 1
    
    @staticmethod
    def makeInferenceOfFeature(rootAttribute: Attribute, featureList: List[str]) -> str:
        if len(rootAttribute.childrenAttributes) == 0: 
            return 'p' if rootAttribute.attributeToSplitOn == -1 else 'e'
        
        inputFeatureLabel: int = featureList[rootAttribute.attributeToSplitOn]
        newRoot: Attribute|None = None
        for childFeatureAttribute in rootAttribute.childrenAttributes:
            if inputFeatureLabel == childFeatureAttribute.label:
                newRoot = childFeatureAttribute
                break
        
        assert newRoot != None, "Failed to find matching label in children" #Shouldn't happen if evertying is correct
        return DescisionTree.makeInferenceOfFeature(newRoot, featureList)

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

def computeAccuracy(dTreeRoot: Attribute, dataPairs: List[tuple[str, List[str]]]) -> float:
    total: int = len(dataPairs)
    correct: int = 0
    for label, featureList in dataPairs:
        dTreeResult = DescisionTree.makeInferenceOfFeature(dTreeRoot, featureList)
        if dTreeResult == label: correct += 1

    return correct/total

def trainDTree(trainData: List[tuple[str, List[str]]], testData: List[tuple[str, List[str]]]) -> None:
    trLabels: List[str] = [x[0] for x in trainData]
    trFeatures: List[List[str]] = [x[1] for x in trainData]

    attr = DescisionTree.createSplits(trLabels, trFeatures) 
    trAc = computeAccuracy(attr, trainData)
    tsAc = computeAccuracy(attr, testData)

    print(f"Training Accuracy {trAc}")
    print(f"Testing Accuracy {tsAc}")

    print("\nDescision Tree")
    print("First value is the label that node corresponds to or \'Top\' if at the root.")
    print("The second value is either the classification or the feature its children's label corresponds to.")
    print(attr.printAttributeTree())

def combineDatasetsAndRetrain(trainData: List[tuple[str, List[str]]], testData: List[tuple[str, List[str]]]) -> None:
    lenOriginalTrain: int = len(trainData)
    copiedCombinedData: List[tuple[str, List[str]]] = copy.deepcopy(trainData) + copy.deepcopy(testData)
    random.shuffle(copiedCombinedData)

    trainDataNew = copiedCombinedData[0:lenOriginalTrain]
    testDataNew = copiedCombinedData[lenOriginalTrain:]

    trainDTree(trainDataNew, testDataNew)
    
def main():
    trainData, testData = getInputData()
    print("Descision Tree using original dataset orderings....")
    trainDTree(trainData, testData)
    print("\n\n\n\nShuffled Train and Test set orders and retrained....")
    combineDatasetsAndRetrain(trainData, testData)

if __name__ == "__main__":
    main()