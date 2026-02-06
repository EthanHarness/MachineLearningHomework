#Ethan Harness (1-30-2026)
#This code is for problem 1 part 2 of homework 1 for CS 6375
#To test the code simply modify main to call homeworkSolutionFunction with input and output data
#Most important code lies in the lossGradient function and startModifiedGradDescent function in the ModifiedGradientDescent class
#Other functions are just to test code
#Inputs initialized based on a uniform distribution using the min and max of the observed values as the function ranges

import random

class ModifiedGradientDescent:
    
    def __init__(self, trainingData, dataLabels, learningRate=.5, iterationCount=100):
        assert len(trainingData) == len(dataLabels), "Length mismatch!"
        
        self.dataSetSize = len(trainingData) #m
        self.inputDimensionSize = len(trainingData[0]) #n
        self.trainingData = trainingData
        self.dataLabels = dataLabels
        self.learningRate = learningRate
        self.iterationCount = iterationCount
        
        min, max = self.getMinAndMaxOfObservedData()
        self.paramsForInput = [random.uniform(min, max) for _ in range(self.inputDimensionSize)] 
        self.paramForConstant = random.uniform(min, max)
         
        self.computeLoss = lambda inData,outData: sum([x*y for x,y in zip(self.paramsForInput, inData)]) + self.paramForConstant - outData
    
    #Gradient is x*sign(loss) with respect to weights and sign(loss) with respect to bias
    #Then need to sum over all data for total loss
    def lossGradient(self):
        assert len(self.trainingData) == len(self.dataLabels)
        
        summedWeightGrad = [0]*self.inputDimensionSize
        summedBiasGrads = 0
        
        for x,y in zip(self.trainingData, self.dataLabels):
            #Since abs(x) is not defined at x = 0, we can use any subgradient on [-1,1]. We use 0.
            sign = 1 if (a:=self.computeLoss(x,y)) > 0 else -1 if a < 0 else 0 
            weightGrads = [sign*a for a in x]
            biasGrad = sign
            
            #Adjusts the cummulative weight bias vals 
            summedWeightGrad = [x + y for x,y in zip(summedWeightGrad, weightGrads)]
            summedBiasGrads += biasGrad
        
        return (summedWeightGrad, summedBiasGrads)
        
    def startModifiedGradDescent(self):
        for _ in range(self.iterationCount):
            paramLoss,biasLoss = self.lossGradient()
            self.paramsForInput = [(x - (self.learningRate*y)) for x,y in zip(self.paramsForInput, paramLoss)]
            self.paramForConstant = self.paramForConstant - (self.learningRate*biasLoss)
            
    #Function to get min and max of observed outputs so we can initialize initial params better
    def getMinAndMaxOfObservedData(self):
        return (min(self.dataLabels), max(self.dataLabels))
    
    #Helper function to compare model to output  
    def testModel(self, showDataDiff):
        absDifference = 0
        for x,y in zip(self.trainingData, self.dataLabels):
            computed = self.computeLoss(x,y)
            absDifference += abs(computed)
            
            if showDataDiff:
                print(self.computeLoss(x,y))
        
        avgDiff = absDifference/self.dataSetSize
        print(f"Average Difference for {self.dataSetSize} observations is {avgDiff}")
        return avgDiff
    
    def getParams(self):
        return self.paramsForInput + [self.paramForConstant]

def runUnitTests():
    changeObservedData = lambda newObserved: (newObserved, len(newObserved[0]), len(newObserved))
    def assertOn1dList(expect, actual, msg):
        for x,y in zip(expect, actual):
            assert x == y, msg
            
    def createCustomRegObj(inData, outData, regWeights, regBias):
        reg = ModifiedGradientDescent(inData, outData)
        reg.paramsForInput = regWeights
        reg.paramForConstant = regBias
        return reg
    
    def testLossComputation():
        #Assume f(a,b) = a + b
        observedData, dims, observedLen = changeObservedData([[1,1],[1,2],[2,1],[2,2]])
        outData = [2,3,3,4]
        errorMsg = "Loss Computation Failure"
        createActual = lambda model,a,b: [model.computeLoss(x,y) for x,y in zip(a,b)]
        
        #Set model params to all 1's
        weights = [1]*dims
        bias = 1
        expected = [1]*observedLen
        actual = createActual(createCustomRegObj(observedData, outData, weights, bias), observedData,outData)
        assertOn1dList(expected, actual, errorMsg)
        
        #Set model paramaters to 1,2,3,... and const to 10
        weights = [x for x in range(1,dims+1)]
        bias = 10
        expected = [11,12,11,12]
        actual = createActual(createCustomRegObj(observedData, outData, weights, bias), observedData, outData)
        assertOn1dList(expected, actual, errorMsg)
        
        #f(x) = x + 2 with all 1's
        observedData, dims, observedLen = changeObservedData([[x] for x in range(5)])
        outData = [x+2 for x in range(5)]
        weights = [1]*dims
        bias = 1
        expected = [-1]*observedLen
        actual = createActual(createCustomRegObj(observedData, outData, weights, bias), observedData, outData)
        assertOn1dList(expected, actual, errorMsg)
        
        #f(x) = x + 2 with 1's for weight and 2 for bias
        observedData, dims, observedLen = changeObservedData([[x] for x in range(5)])
        outData = [x+2 for x in range(5)]
        weights = [1]*dims
        bias = 2
        expected = [0]*observedLen
        actual = createActual(createCustomRegObj(observedData, outData, weights, bias), observedData, outData)
        assertOn1dList(expected, actual, errorMsg)
        
    def testIndividualPointGradientComputation():
        #Assume f(a,b) = a + b
        observedData, dims, observedLen = changeObservedData([[1,1]])
        outData = [2]
        errorMsg = "Individual Gradient Computation Failure"
        
        #Make weights all 1 and bias 2
        weights = [1]*dims
        bias = 2
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        expectedWeight = [1,1]
        expectedBias = 1
        assert actualBias == expectedBias, errorMsg
        assertOn1dList(expectedWeight, actualWeight, errorMsg)
        
        #Make weights all 1 and bias -1
        weights = [1]*dims
        bias = -1
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        expectedWeight = [-1,-1]
        expectedBias = -1
        assert actualBias == expectedBias, errorMsg
        assertOn1dList(expectedWeight, actualWeight, errorMsg)
        
        #Make weights all 1 and bias 0
        weights = [1]*dims
        bias = 0
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        expectedWeight = [0,0]
        expectedBias = 0
        assert actualBias == expectedBias, errorMsg
        assertOn1dList(expectedWeight, actualWeight, errorMsg)
        
        #Make weights 1,2,... and bias 0
        weights = [x+1 for x in range(dims)]
        bias = 0
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        expectedWeight = [1,1]
        expectedBias = 1
        assert actualBias == expectedBias, errorMsg
        assertOn1dList(expectedWeight, actualWeight, errorMsg)
        
        #Make weights 1,2,... and change the data to another point
        observedData, dims, observedLen = changeObservedData([[3,-1,-7]])
        outData = [-5]
        weights = [x+1 for x in range(dims)]
        bias = 0
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        expectedWeight = [-3,1,7]
        expectedBias = -1
        assert actualBias == expectedBias, errorMsg
        assertOn1dList(expectedWeight, actualWeight, errorMsg)
        
        #Change data again
        observedData, dims, observedLen = changeObservedData([[3]])
        outData = [3]
        weights = [x+1 for x in range(dims)]
        bias = 0
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        expectedWeight = [0]
        expectedBias = 0
        assert actualBias == expectedBias, errorMsg
        assertOn1dList(expectedWeight, actualWeight, errorMsg)
        
    def testGradientComputation():
        errorMsg = "Cumulative Gradient Computation Failure"
        
        #f(a,b) = a + b
        observedData, dims, observedLen = changeObservedData([[1,2],[2,3]])
        outData = [3,5]
        weights = [1]*dims
        bias = 1
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        expectedWeight = [3,5]
        expectedBias = 2
        assert actualBias == expectedBias, errorMsg
        assertOn1dList(expectedWeight, actualWeight, errorMsg)
        
        #Under test
        bias = -1
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        expectedWeight = [-3,-5]
        expectedBias = -2
        assert actualBias == expectedBias, errorMsg
        assertOn1dList(expectedWeight, actualWeight, errorMsg)
        
        #Good test
        bias = 0
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        expectedWeight = [0,0]
        expectedBias = 0
        assert actualBias == expectedBias, errorMsg
        assertOn1dList(expectedWeight, actualWeight, errorMsg)
        
        #Adjust reg weights
        observedData, dims, observedLen = changeObservedData([[1,2],[3,1]])
        outData = [3,4]
        weights = [-5,5]
        bias = 0
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        expectedWeight = [-2,1]
        expectedBias = 0
        assert actualBias == expectedBias, errorMsg
        assertOn1dList(expectedWeight, actualWeight, errorMsg)
        
            
    testLossComputation()
    testIndividualPointGradientComputation()
    testGradientComputation()
    print("Unit tests complete")
    
    
      

#This function adheres to the input output information as specified in the homework
#Takes in observation input data matrix and observation output vector and runs gradient descent according to the loss function in problem 2
#Has learning rate of .5 and runs for 100 iterations by default
#You can pass in a different learning rate and number of iterations to customize this
def homeworkSolutionFunction(observationInputMatrix, observationOutputVector):
    descent = ModifiedGradientDescent(observationInputMatrix, observationOutputVector)
    return descent.getParams()

#Helper code to test gradient descent code           
def testHelper():
    inputGenerator = lambda numInputs, rg, numPoints: [[random.uniform(rg*-1, rg) for _ in range(numInputs)] for _ in range(numPoints)]
    generateXObservationData = lambda func, points, noise: [[x, a:=func(x), a+random.uniform(noise*-1, noise)] for x in points]

    functionToLearn = lambda inList: sum([(x**3) - (11*(x**2)) - (5*x) + 22  for x in inList])
    LEARN_FUNCTION_INPUTS_CONSTANT = 5
    OBSERVATION_RANGE_CONSTANT = 100
    NUMBER_OF_OBSERVATIONS = 100
    NOISE_CONSTANT = 50


    data = generateXObservationData(functionToLearn, inputGenerator(LEARN_FUNCTION_INPUTS_CONSTANT, 
                                                             OBSERVATION_RANGE_CONSTANT, NUMBER_OF_OBSERVATIONS), NOISE_CONSTANT)
    learningRate = .5
    epochs = 1000
    observationData = [x[0] for x in data]
    outputData = [x[2] for x in data]
    showDataDiff = False
    
    reg = ModifiedGradientDescent(observationData, outputData, learningRate, epochs)
    print("\n\nInitial Test")
    initError = reg.testModel(showDataDiff)
    reg.startModifiedGradDescent()
    
    print("\nFinal Test")
    finalError = reg.testModel(showDataDiff)
    
    print(f"\nError Decreased By: {initError - finalError}")


def main():
    runUnitTests()
    #testHelper() #Can remove and call homeworkSolutionFunction to test as outlined by problem description

if __name__ == "__main__":
    main()
    