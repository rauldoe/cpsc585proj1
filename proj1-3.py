
from string import ascii_lowercase
from random import random
from dataset import TRAINING_DATA, TEST_DATA

g_neuralNetworkSet = None

def char2vec(char):
    return [
        -1 if pixel == '.' else 1
        for line in char
        for pixel in line
    ]

def initTrainingDataInfo(trainingData, ids):
    trainingDataInfo = {}
    for i in range(len(trainingData)):
        training = trainingData[i]
        cid = ids[i]
        trainingDataInfo[cid] = char2vec(training)

    return trainingDataInfo

def initTestDataInfo(testData, ids):
    testDataInfo = {}
    for i in range(len(testData)):
        test = testData[i]
        cid = ids[i]
        testDataInfo[cid] = char2vec(test)

    return testDataInfo

def initWeights(inputCount):
    # must account for bias
    return [random() for i in range(inputCount+1)]

def getInputCount(trainingData):
    return len(trainingData[0])

def initNeuralNetwork(id, trainingData):
    inputCount = getInputCount(trainingData)

    weights = initWeights(inputCount)
    trainingModel = {}
    for key, value in trainingData.items():
        trainingModel[key] = {'id' : key, 'vector' : value, 'expected' : 1 if key==id else 0}
    
    return {'weights' : weights, 'model' : trainingModel}

def train(trainingDataInfo):
    print('train')

    return neuralNetwork

def predict(neuralNetwork, testDataInfo):
    print('predict')

    return predictHistory

def runNeuralNetwork(trainingDataInfo, testDataInfo):
    print('Run Neural Network')

    neuralNetwork = train(trainingDataInfo)
    predict(neuralNetwork, testDataInfo)

def main():
    print('main')

if __name__ == '__main__':
    main()