#THIS IS ONLY FOR TESTING PURPOSES 
#THE MAIN LOGIC DOES NOT TOUCH THIS FILE AT ALL
#IF YOU ARE GRADING THIS IGNORE THIS FILE

from typing import List

from sklearn import tree
from matplotlib import pyplot as plt

from problem1Main import getInputData
from descisionTree import Attribute

def convertFeatureList(featureList: List[str]) -> List[int]:
    def convertFeatureToInt(featureNumber: int, val: str) -> int:
        domain = Attribute.getDomainOfFeature(featureNumber)
        for index,domainValue in enumerate(domain):
            if domainValue == val: return index

    return [convertFeatureToInt(index, val) for index,val in enumerate(featureList)]
        

def main():
    trainData, testData = getInputData()

    trData = [convertFeatureList(x[1]) for x in trainData]
    trLabel = [1 if x[0] == 'p' else -1 for x in trainData]

    tsData = [convertFeatureList(x[1]) for x in testData]
    tsLabel = [1 if x[0] == 'p' else -1 for x in testData]

    clf = tree.DecisionTreeClassifier().fit(trData, trLabel)
    resOnTrain = clf.predict(trData)
    resOnTest = clf.predict(tsData)
    
    for x,y in zip(resOnTrain, trLabel):
        if x != y: print(False)

    for x,y in zip(resOnTest, tsLabel):
        if x != y: print(False)

    print(clf.get_depth())
    print(clf.get_n_leaves())

    testSamples: List[int] = [567,681]
    samples: List[List[int]] = []
    for index in testSamples:
        samples.append(convertFeatureList(trData[index]))
    print(clf.decision_path(samples))

    tree.plot_tree(clf, proportion=True)
    plt.show()

if __name__ == "__main__":
    main()