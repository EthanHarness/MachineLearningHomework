import copy

class PerceptronAlgorithm:
    
    def __init__(self, trainingData, dataLabels):
        assert len(trainingData) == len(dataLabels), "Length mismatch!"
        self.transformInput(trainingData)
        
        self.dataSetSize = len(trainingData) #m
        self.inputDimensionSize = len(trainingData[0]) #n
        self.trainingData = trainingData
        self.dataLabels = dataLabels
        
        self.paramsForInput = [0]*self.inputDimensionSize
        self.paramForConstant = 0
         
        self.computeMax2Argument = lambda inData,outData: (sum([x*y for x,y in zip(self.paramsForInput, inData)]) + self.paramForConstant)*(outData*-1)
        self.classify = lambda inData, outData: max(self.computeMax2Argument(inData, outData), 0)

    def transformInput(self, trainingData):
        for list in trainingData:
            #Problem explicitly states feature map takes point in R2 and goes to a point in R3
            assert len(list) == 2, "Violates Problem Description"
            list.append((list[0]**2) + (list[1]**2))
            
    def computeLossGradient(self):
        assert len(self.trainingData) == len(self.dataLabels)
        
        summedWeightGrads = [0]*self.inputDimensionSize
        summedBiasGrad = 0
        
        for x,y in zip(self.trainingData, self.dataLabels):
            correctFlag = True if self.computeMax2Argument(x,y) < 0 else False
            weightGrads = [-1*a*y for a in x] if not correctFlag else [0]*self.inputDimensionSize
            biasGrad = -1*y if not correctFlag else 0
            
            summedWeightGrads = [x + y for x,y in zip(summedWeightGrads, weightGrads)]
            summedBiasGrad += biasGrad
        
        summedWeightGrads = [x/self.dataSetSize for x in summedWeightGrads]
        summedBiasGrad = summedBiasGrad/self.dataSetSize
            
        return (summedWeightGrads, summedBiasGrad)
    
    #States to iterate in order so iteration will be used to keep track of where we are 
    def computeLossGradientStochastic(self, iteration):
        assert len(self.trainingData) == len(self.dataLabels)
        position = iteration % self.dataSetSize
        
        inData = self.trainingData[position]
        outData = self.dataLabels[position]
        
        correctFlag = True if self.computeMax2Argument(inData, outData) < 0 else False
        weightGrads = [-1*a*outData for a in inData] if not correctFlag else [0]*self.inputDimensionSize
        biasGrad = -1*outData if not correctFlag else 0
        
        return (weightGrads, biasGrad)
        
    def computeLoss(self):
        error = 0
        for x,y in zip(self.trainingData, self.dataLabels):
            error += self.classify(x,y)
        return error/self.dataSetSize
                
    #Alpha fixed to some value and can use either stochastic gradient or normal for problem 2 part 3 subsection 3 
    def varyAlphaGradDescentFunc(self, maxIterations=10000, reportIterations=None, alpha=1, stochastic=False):
        error = self.computeLoss()
        iteration = 1
        reportIterations = reportIterations if reportIterations != None else []
        while (iteration <= maxIterations):
            paramLoss, biasLoss = self.computeLossGradient() if not stochastic else self.computeLossGradientStochastic(iteration)
            self.paramsForInput = [(x - (alpha*y)) for x,y in zip(self.paramsForInput, paramLoss)]
            self.paramForConstant = self.paramForConstant - (alpha*biasLoss)
            error = self.computeLoss()
            
            if iteration in reportIterations:
                print(f"Iteration {iteration} Results")
                print(f"W Values: {self.paramsForInput}")
                print(f"B Value: {self.paramForConstant}")
                print(f"Loss Value: {error}")
                
            iteration += 1
                
        
def processInput():
    inputData = []
    inputDataLabels = []
    with open("./perceptron.data", "r") as file:
        for line in file:
            valArr = line.split(",")
            inputData.append([float(valArr[0]), float(valArr[1])])
            inputDataLabels.append(int(valArr[2]))
            
    return (inputData, inputDataLabels)
    
MAX_ITERATIONS_CONSTANT = 100000
REPORT_ARRAY_CONSTANT = [1,10,100,1000,10000,100000]     

def problem2Part3Subsection1():
    data, dataLabels = processInput()
    obj = PerceptronAlgorithm(data, dataLabels)
    print("Subsection 1........")
    obj.varyAlphaGradDescentFunc(MAX_ITERATIONS_CONSTANT, REPORT_ARRAY_CONSTANT, 1, False)
    print("\n")
    
def problem2Part3Subsection2():
    data, dataLabels = processInput()
    obj = PerceptronAlgorithm(data, dataLabels)
    print("Subsection 2........")
    obj.varyAlphaGradDescentFunc(MAX_ITERATIONS_CONSTANT, REPORT_ARRAY_CONSTANT, 1, True)
    print("\n")
    
def problem2Part3Subsection3(rangeConstant=3):
    print("Subsection 3........")
    data, dataLabels = processInput()
    
    for x in range(rangeConstant*2):
        alpha = ((x%rangeConstant)+1)/(rangeConstant)
        stochastic = x >= rangeConstant
        print(f"Alpha fixed to {alpha} and stochastic={stochastic}")
        
        dataCopy = copy.deepcopy(data)
        labelCopy = copy.deepcopy(dataLabels)
        obj = PerceptronAlgorithm(dataCopy, labelCopy)
        obj.varyAlphaGradDescentFunc(MAX_ITERATIONS_CONSTANT, REPORT_ARRAY_CONSTANT, alpha, stochastic)
        print("")

def main():
    problem2Part3Subsection1()
    problem2Part3Subsection2()
    problem2Part3Subsection3(3)

if __name__ == "__main__":
    main()