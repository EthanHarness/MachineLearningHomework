from typing import List
import math

CAP_SHAPE_DOMAIN = ['b','c','x','f','k','s']
CAP_SURFACE_DOMAIN = ['f','g','y','s']
CAP_COLOR_DOMAIN = ['n','b','c','g','r','p','u','e','w','y']
BRUISES_DOMAIN = ['t','f']
ODOR_DOMAIN = ['a','l','c','y','f','m','n','p','s']
GILL_ATTCH_DOMAIN = ['a','f','d','n']
GILL_SPACE_DOMAIN = ['c','w','d']
GILL_SIZE_DOMAIN = ['b','n']
GILL_COLOR_DOMAIN = ['k','n','b','h','g','r','o','p','u','e','w','y']
STALK_SHAPE_DOMAIN = ['e','t']
STALK_ROOT_DOMAIN = ['b','c','u','e','z','r', 'm'] 
STALK_SURFACE_ABOVE_DOMAIN = ['f','y','k','s']
STALK_SURFACE_BELOW_DOMAIN = ['f','y','k','s']
STALK_COLOR_ABOVE_DOMAIN = ['n','b','c','g','o','p','e','w','y']
STALK_COLOR_BELOW_DOMAIN = ['n','b','c','g','o','p','e','w','y']
VEIL_TYPE_DOMAIN = ['p','u']
VEIL_COLOR_DOMAIN = ['n','o','w','y']
RING_NUMBER_DOMAIN = ['n','o','t']
RING_TYPE_DOMAIN = ['c','e','f','l','n','p','s','z',]
SPORE_PRIINT_DOMAIN = ['k','n','b','h','r','o','u','w','y']
POPULATION_DOMAIN = ['a','c','n','s','v','y']
HABITAT_DOMAIN = ['g','l','m','p','u','w','d']

LABEL_DOMAIN = ['p','e']
NUMBER_OF_FEATURES = 22
LOG_BASE = 10


class Attribute:
    def __init__(self, featureNumber: int, label: str = "Top"):
        self.attributeToSplitOn = featureNumber
        self.label = label
        self.childrenAttributes: List[Attribute] = []

    @staticmethod
    def getDomainOfFeature(featureNumber: int):
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

    def addAttributeSplitToChildren(self, attr):
        self.childrenAttributes.append(attr)

    def printAttributeTree(self, numTabs: int = 0):
        printVal = self.attributeToSplitOn
        if self.attributeToSplitOn == -1: printVal = 'p'
        elif self.attributeToSplitOn == -2: printVal = 'e'

        tabStringP = "\t" * numTabs
        resString = tabStringP + self.label + " " + str(printVal) + "\n"
        for x in self.childrenAttributes:
            resString += x.printAttributeTree(numTabs+1)
        return resString
    
    def addClassLabel(self, label: str):
        self.label = label

class DescisionTree:
    @staticmethod
    def createSplits(label: List[str], featureList: List[List[str]]):
        if len(label) == 1 or DescisionTree.labelsHave1Val(label): return Attribute(-1 if label[0] == 'p' else -2, label[0])

        splitAttr = DescisionTree.findAttributeToSplitOn(label, featureList)
        splitDomain = Attribute.getDomainOfFeature(splitAttr)
        attr = Attribute(splitAttr)

        divisions = {x:[[],[]] for x in splitDomain}
        for l, fList in zip(label, featureList):
            divisions[fList[splitAttr]][0].append(l) 
            divisions[fList[splitAttr]][1].append(fList)
        
        for key in divisions.keys():
            if len(divisions[key][0]) == 0: continue
            attrChild = DescisionTree.createSplits(divisions[key][0], divisions[key][1])
            attr.addAttributeSplitToChildren(attrChild)
            attrChild.addClassLabel(key)

        return attr
            

    @staticmethod
    def entropyPoisonous(labels: List[str]):
        poisonSamples = len([x for x in labels if x == 'p'])
        samples = len(labels)
        if (poisonSamples == 0 or poisonSamples == samples): return 0

        p = poisonSamples / samples
        q = 1 - p

        return -1*((p*math.log(p,LOG_BASE)) + (q*math.log(q,LOG_BASE)))
    
    @staticmethod
    def entropyGivenAttribute(labels: List[str], featureList: List[List[str]], featureNumber: int):
        domain = Attribute.getDomainOfFeature(featureNumber)
        entropy = 0
        samples = len(labels)

        for attribute in domain:
            attrCount = len([1 for x in featureList if x[featureNumber] == attribute])

            if attrCount == 0: continue

            pXeqlx = attrCount / samples
            cummulativeSum = 0
            for label in LABEL_DOMAIN:
                condCount = len([1 for labelVal,features in zip(labels, featureList) if labelVal == label and features[featureNumber] == attribute])

                if condCount == 0: continue

                pYGivenX = condCount/attrCount
                logpYGivenX = math.log(pYGivenX, LOG_BASE)
                cummulativeSum += (pYGivenX*logpYGivenX)
            entropy += cummulativeSum*pXeqlx

        return entropy*-1
    
    @staticmethod
    def findInformationGainForAttribute(labels: List[str], featureList: List[List[str]], featureNumber: int):
        condEntropy = DescisionTree.entropyGivenAttribute(labels, featureList, featureNumber)
        entropyPoisonous = DescisionTree.entropyPoisonous(labels)
        return entropyPoisonous - condEntropy
    
    @staticmethod
    def findAttributeToSplitOn(labels: List[str], featureList: List[List[str]]):
        
        maxInformationGain = 0
        featureToSplitOn = 0
        for x in range(NUMBER_OF_FEATURES):
            ig = DescisionTree.findInformationGainForAttribute(labels, featureList, x)
            if ig >= maxInformationGain:
                maxInformationGain = ig
                featureToSplitOn = x

        return featureToSplitOn
    
    @staticmethod
    def labelsHave1Val(labels: List[str]):
        return len(set(labels)) == 1
    
    @staticmethod
    def makeInferenceOfFeature(rootAttribute: Attribute, featureList: List[str]):
        if len(rootAttribute.childrenAttributes) == 0: 
            return 'p' if rootAttribute.attributeToSplitOn == -1 else 'e'
        
        inputFeatureLabel = featureList[rootAttribute.attributeToSplitOn]
        newRoot = None
        for childFeatureAttribute in rootAttribute.childrenAttributes:
            if inputFeatureLabel == childFeatureAttribute.label:
                newRoot = childFeatureAttribute
                break
        
        assert newRoot != None, "Failed to find matching label in children" #Shouldn't happen if domains are correct
        return DescisionTree.makeInferenceOfFeature(newRoot, featureList)


def makeAssertionOnInput(featureList: List[str]):
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

def computeAccuracy(dTreeRoot, dataset, labels):
    total = len(labels)
    correct = 0
    for featureList, label in zip(dataset, labels):
        dTreeResult = DescisionTree.makeInferenceOfFeature(dTreeRoot, featureList)
        if dTreeResult == label: correct += 1

    return correct/total

def splitOut(input: List[tuple[str, List[str]]]) -> tuple[List[str], List[List[str]]]:
    labels = [x[0] for x in input]
    features = [x[1] for x in input]
    return labels, features

def main():
    trainData, testData = getInputData()
    trLabels, trFeatures = splitOut(trainData)
    tsLabels, tsFeatures = splitOut(testData)

    attr = DescisionTree.createSplits(trLabels, trFeatures) 
    
    trAc = computeAccuracy(attr, trFeatures, trLabels)
    tsAc = computeAccuracy(attr, tsFeatures, tsLabels)

    print(f"Training Accuracy {trAc}")
    print(f"Testing Accuracy {tsAc}")

if __name__ == "__main__":
    main()