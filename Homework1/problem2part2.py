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
        
        self.paramsForInput = [random.uniform(-.01, -.01) for _ in range(self.inputDimensionSize)] 
        self.paramForConstant = self.computeMedianOfLabels()
         
        self.computeLoss = lambda inData,outData: sum([x*y for x,y in zip(self.paramsForInput, inData)]) + self.paramForConstant - outData
        self.computeOutput = lambda inData: sum([x*y for x,y in zip(self.paramsForInput, inData)]) + self.paramForConstant
    
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
        
    def startModifiedGradDescent(self, evaluatePerIteration=False):
        for a in range(self.iterationCount):
            paramLoss,biasLoss = self.lossGradient()
            self.paramsForInput = [(x - (self.learningRate*y)) for x,y in zip(self.paramsForInput, paramLoss)]
            self.paramForConstant = self.paramForConstant - (self.learningRate*biasLoss)
            
            if evaluatePerIteration: self.modelEvaluation(self.getAvgDifference(), a)
            
    def performGradientDescentToConvergence(self, cvgThreashold=5, iterationCutoff=10000, evaluatePerIteration=False, dynamicLRAjustment=True):
        avgDifference = self.getAvgDifference()
        iteration = 0
        while(avgDifference > cvgThreashold and iteration < iterationCutoff):
            paramLoss,biasLoss = self.lossGradient()
            alpha = self.learningRate if not dynamicLRAjustment else (1/(1+iteration))
            self.paramsForInput = [(x - (alpha*y)) for x,y in zip(self.paramsForInput, paramLoss)]
            self.paramForConstant = self.paramForConstant - (alpha*biasLoss)
            iteration += 1
            avgDifference = self.getAvgDifference()
            
            if evaluatePerIteration: self.modelEvaluation(avgDifference, iteration)
    
    def computeMedianOfLabels(self):
        middle = len(self.dataLabels)//2
        copy = self.dataLabels.copy()
        copy.sort()
        return copy[middle]
        
    
    def getAvgDifference(self):
        absDifference = 0
        for x,y in zip(self.trainingData, self.dataLabels):
            absDifference += abs(self.computeLoss(x,y))
        
        return absDifference/self.dataSetSize
     
    def modelEvaluation(self, avgDiff, iterationNumber=None):
        if iterationNumber == None:
            print(f"Average Difference for {self.dataSetSize} observations is {avgDiff}")
        else: 
            print(f"Average Difference for {self.dataSetSize} observations is {avgDiff} for iteration {iterationNumber}")
    
    def getParams(self):
        return self.paramsForInput + [self.paramForConstant]

#This function just tests the core functions of ModifiedGradientDescent and can be ignored
def runUnitTests():
    def assertOn1dList(expect, actual, msg):
        for x,y in zip(expect, actual):
            assert x == y, msg
            
    def assertWeightsBias(eWeights, aWeights, eBias, aBias, msg):
        assertOn1dList(eWeights, aWeights, msg)
        assert eBias == aBias, msg
    
    changeObservedData = lambda newObservedIn, newOut: (newObservedIn, newOut, len(newObservedIn[0]), len(newObservedIn))
    changeWeightsBias = lambda weights, bias: (weights, bias)
    changeExpected = lambda eWeights, eBias=None: (eWeights, eBias)
    changeRegParams = lambda alpha, iterations=None: (alpha, iterations)
            
    def createCustomRegObj(inData, outData, regWeights, regBias, alpha=None, iterations=None):
        if iterations is None: iterations = 1
        reg = ModifiedGradientDescent(inData, outData, alpha, iterations)
        reg.paramsForInput = regWeights
        reg.paramForConstant = regBias
        return reg
    
    def testLossComputation():
        errorMsg = "Loss Computation Failure"
        
        #Assume f(a,b) = a + b
        observedData, outData, dims, observedLen = changeObservedData([[1,1],[1,2],[2,1],[2,2]], [2,3,3,4])
        createActual = lambda model,a,b: [model.computeLoss(x,y) for x,y in zip(a,b)]
        
        #Set model params to all 1's
        weights, bias = changeWeightsBias([1]*dims, 1)
        expected, _ = changeExpected([1]*observedLen)
        actual = createActual(createCustomRegObj(observedData, outData, weights, bias), observedData,outData)
        assertOn1dList(expected, actual, errorMsg)
        
        #Set model paramaters to 1,2,3,... and const to 10
        weights, bias = changeWeightsBias([x for x in range(1,dims+1)], 10)
        expected, _ = changeExpected([11,12,11,12])
        actual = createActual(createCustomRegObj(observedData, outData, weights, bias), observedData, outData)
        assertOn1dList(expected, actual, errorMsg)
        
        #f(x) = x + 2 with all 1's
        observedData, outData, dims, observedLen = changeObservedData([[x] for x in range(5)], [x+2 for x in range(5)])
        weights, bias = changeWeightsBias([1]*dims, 1)
        expected, _ = changeExpected([-1]*observedLen)
        actual = createActual(createCustomRegObj(observedData, outData, weights, bias), observedData, outData)
        assertOn1dList(expected, actual, errorMsg)
        
        #f(x) = x + 2 with 1's for weight and 2 for bias
        weights, bias = changeWeightsBias(weights, 2)
        expected, _ = changeExpected([0]*observedLen)
        actual = createActual(createCustomRegObj(observedData, outData, weights, bias), observedData, outData)
        assertOn1dList(expected, actual, errorMsg)
        
    def testIndividualPointGradientComputation():
        errorMsg = "Individual Gradient Computation Failure"
        
        #Assume f(a,b) = a + b #Make weights all 1 and bias 2
        observedData, outData, dims, _ = changeObservedData([[1,1]], [2])
        weights, bias = changeWeightsBias([1]*dims, 2)
        expectedWeight, expectedBias = changeExpected([1,1], 1)
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        assertWeightsBias(expectedWeight, actualWeight, expectedBias, actualBias, errorMsg)
        
        #Make weights all 1 and bias -1
        weights, bias = changeWeightsBias(weights, -1)
        expectedWeight, expectedBias = changeExpected([-1,-1], -1)
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        assertWeightsBias(expectedWeight, actualWeight, expectedBias, actualBias, errorMsg)
        
        #Make weights all 1 and bias 0
        weights, bias = changeWeightsBias(weights, 0)
        expectedWeight, expectedBias = changeExpected([0,0],0)
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        assertWeightsBias(expectedWeight, actualWeight, expectedBias, actualBias, errorMsg)
        
        #Make weights 1,2,... and bias 0
        weights, bias = changeWeightsBias([x+1 for x in range(dims)], 0)
        expectedWeight, expectedBias = changeExpected([1,1], 1)
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        assertWeightsBias(expectedWeight, actualWeight, expectedBias, actualBias, errorMsg)
        
        #Make weights 1,2,... and change the data to another point
        observedData, outData, dims, _ = changeObservedData([[3,-1,-7]], [-5])
        weights, bias = changeWeightsBias([x+1 for x in range(dims)], 0)
        expectedWeight, expectedBias = changeExpected([-3,1,7], -1)
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        assertWeightsBias(expectedWeight, actualWeight, expectedBias, actualBias, errorMsg)
        
        #Change data again
        observedData, outData, dims, _ = changeObservedData([[3]], [3])
        weights, bias = changeWeightsBias([x+1 for x in range(dims)], 0)
        expectedWeight, expectedBias = changeExpected([0],0)
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        assertWeightsBias(expectedWeight, actualWeight, expectedBias, actualBias, errorMsg)
        
    def testGradientComputation():
        errorMsg = "Cumulative Gradient Computation Failure"
        
        #f(a,b) = a + b
        observedData, outData, dims, _ = changeObservedData([[1,2],[2,3]], [3,5])
        weights, bias = changeWeightsBias([1]*dims, 1)
        expectedWeight, expectedBias = changeExpected([3,5], 2)
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        assertWeightsBias(expectedWeight, actualWeight, expectedBias, actualBias, errorMsg)
        
        #Under test
        weights, bias = changeWeightsBias(weights, -1)
        expectedWeight, expectedBias = changeExpected([-3,-5], -2)
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        assertWeightsBias(expectedWeight, actualWeight, expectedBias, actualBias, errorMsg)
        
        #Good test
        weights, bias = changeWeightsBias(weights, 0)
        expectedWeight, expectedBias = changeExpected([0,0], 0)
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        assertWeightsBias(expectedWeight, actualWeight, expectedBias, actualBias, errorMsg)
        
        #Adjust reg weights
        observedData, outData, dims, _ = changeObservedData([[1,2],[3,1]], [3,4])
        weights, bias = changeWeightsBias([-5,5], 0)
        expectedWeight, expectedBias = changeExpected([-2,1], 0)
        actualWeight, actualBias = createCustomRegObj(observedData, outData, weights, bias).lossGradient()
        assertWeightsBias(expectedWeight, actualWeight, expectedBias, actualBias, errorMsg)
           
    def testGradientUpdate():
        errorMsg = "Gradient Update Failure"
        
        def updateParams(regObj, weights, bias, alpha, iterations):
            adjWeights = regObj.paramsForInput
            adjBias = regObj.paramForConstant
            for _ in range(iterations):
                gradWeights, gradBias = regObj.lossGradient()
                adjWeights = [y - (alpha*x) for x,y in zip(gradWeights, weights)]
                adjBias = bias - (alpha*gradBias)
            return (adjWeights, adjBias)
        
        def getActuals(regObj):
            regObj.startModifiedGradDescent()
            return (regObj.paramsForInput, regObj.paramForConstant)
        
        #f(a+b) 1 iteration and alpha=.5
        observedData, outData, dims, _ = changeObservedData([[1,1]], [2])
        weights, bias = changeWeightsBias([1]*dims, 2)
        alpha, iterations = changeRegParams(.5, 1)
        regObject = createCustomRegObj(observedData, outData, weights.copy(), bias, alpha)
        expectedWeights, expectedBias = updateParams(regObject, weights, bias, alpha, iterations)
        actualWeights, actualBias = getActuals(regObject)
        assertWeightsBias(expectedWeights, actualWeights, expectedBias, actualBias, errorMsg)
        
        #1 iteration and alpha=.75
        alpha, iterations = changeRegParams(.75, iterations)
        regObject = createCustomRegObj(observedData, outData, weights.copy(), bias, alpha)
        expectedWeights, expectedBias = updateParams(regObject, weights, bias, alpha, iterations)
        actualWeights, actualBias = getActuals(regObject)
        assertWeightsBias(expectedWeights, actualWeights, expectedBias, actualBias, errorMsg)
        
        #2 iterations and alpha=.75
        alpha, iterations = changeRegParams(alpha, 2)
        regObject = createCustomRegObj(observedData, outData, weights.copy(), bias, alpha)
        expectedWeights, expectedBias = updateParams(regObject, weights, bias, alpha, iterations)
        actualWeights, actualBias = getActuals(regObject)
        assertWeightsBias(expectedWeights, actualWeights, expectedBias, actualBias, errorMsg)
        
    testLossComputation()
    testIndividualPointGradientComputation()
    testGradientComputation()
    testGradientUpdate()
    print("Unit tests complete")
      

#This function adheres to the input output information as specified in the homework
#Takes in observation input data matrix and observation output vector and runs gradient descent according to the loss function in problem 2
#This performs gradient descent to convergence which is defined as the average loss of the model is less then 5
#If the model does not reach this after 10k iterations, it will terminate early. 
#You can modify this behavior by passing in arguments into the function
def homeworkSolutionFunction(observationInputMatrix, observationOutputVector):
    descent = ModifiedGradientDescent(observationInputMatrix, observationOutputVector)
    descent.performGradientDescentToConvergence()
    return descent.getParams()

#Helper code to test gradient descent code           
def testHelper():
    inputGenerator = lambda numInputs, rg, numPoints: [[random.uniform(rg*-1, rg) for _ in range(numInputs)] for _ in range(numPoints)]
    generateXObservationData = lambda func, points, noise: [[x, a:=func(x), a+random.uniform(noise*-1, noise)] for x in points]
    
    def functionToLearn(inList):
        return sum([5*x + 127 for x in inList])

    LEARN_FUNCTION_INPUTS_CONSTANT = 3
    OBSERVATION_RANGE_CONSTANT = 10000
    NUMBER_OF_OBSERVATIONS = 100
    NOISE_CONSTANT = 0
    learningRate = .5
    epochs = 2000
    cvgThreashold = 5
    maxIterations = 1000000


    data = generateXObservationData(functionToLearn, inputGenerator(LEARN_FUNCTION_INPUTS_CONSTANT, 
                                                             OBSERVATION_RANGE_CONSTANT, NUMBER_OF_OBSERVATIONS), NOISE_CONSTANT)
    observationData = [x[0] for x in data]
    outputData = [x[2] for x in data]
    
    reg = ModifiedGradientDescent(observationData, outputData, learningRate, epochs)
    initError = sum([abs(reg.computeOutput(observationData[x])) for x in range(NUMBER_OF_OBSERVATIONS)])
    print(f"Initial Error {initError}")
    
    reg.performGradientDescentToConvergence(cvgThreashold, maxIterations, True, True)
    
    finalError = sum([abs(reg.computeOutput(observationData[x])) for x in range(NUMBER_OF_OBSERVATIONS)])
    print(f"Final Error {finalError}")
    print(f"Error Decreased By: {finalError - initError}\n\n")
    for x in range(NUMBER_OF_OBSERVATIONS):
        print(functionToLearn(observationData[x]), reg.computeOutput(observationData[x]))


def main():
    runUnitTests()
    testHelper() #Can remove and call homeworkSolutionFunction to test as outlined by problem description

if __name__ == "__main__":
    main()
    