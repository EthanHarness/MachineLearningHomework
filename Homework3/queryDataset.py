from typing import List

from descisionTree import Attribute

class AttributeValueWrapper:
    def __init__(self, attributeNumber: int, attributeValue: str) -> None:
        assert attributeValue in Attribute.getDomainOfFeature(attributeNumber), "Invalid input. Attribute value not member of attribute number domain"
        self.attrNum: int = attributeNumber
        self.attrVal: str = attributeValue

    def isFeatureValid(self, featureList: List[str]) -> bool:
        if featureList[self.attrNum] == self.attrVal: return True
        return False

class QuerySegment: 
    def __init__(self, attrWrapperList: List[AttributeValueWrapper]) -> None:
        self.attrs: List[AttributeValueWrapper] = attrWrapperList

    def isFeatureValid(self, feature: List[str]) -> bool:
        for x in self.attrs:
            if not x.isFeatureValid(feature): return False
        return True
    
    def getIndexesOfValidFeatures(self, fList: List[List[str]]) -> List[int]:
        indexes: List[int] = []
        for index,feature in enumerate(fList):
            if self.isFeatureValid(feature): indexes.append(index)
        return indexes
    

def prettyPrintRes(res: List[tuple[int, List[str]]]):
    headerString = " "
    for x in range(10):
        headerString += f" {x}   "
    for x in range(10, 22):
        headerString += f"{x}   "
    print(headerString)
    for index,x in res:
        print(x, index)

#Query functions just to test things and make sure dTree is correct
#Treats each segment as one giant "and" then after all segments execute "or" the results together 
def queryDataUsingQuerySegments(listOfSegments: List[QuerySegment], dataset: List[tuple[str, List[str]]]) -> List[tuple[int, List[str]]]:
    listOfFeatures: List[List[str]] = [x[1] for x in dataset]
    setValidIndexes: set[int] = set()
    for segment in listOfSegments:
        setValidIndexes.update(segment.getIndexesOfValidFeatures(listOfFeatures))
    return [(index,x) for index,x in enumerate(listOfFeatures) if index in setValidIndexes]