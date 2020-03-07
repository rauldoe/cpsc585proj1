#TODO:
#1. Attach correct values to test data -> pd
#2. Update weights and bias to correct the passes -> pd
#3. Matplotlib to plot the error rate as a funciton of the number of epocs -> ken, william
#4. Q. Are all letters in the training set linearly separable? -> william
#5. Q. Are all letters in the test set correctly classified? -> pd
#6. Q. How similar are the misclassified items to the items in the training set? -> ken, william
#7. Fix train, test for all characters -> pd

#!/usr/bin/env python3

# Character recognition
#
# See Fausett, Example 2.8, pp. 55-56
#

from string import ascii_lowercase
from random import random
from dataset1 import *

NULL_ID = None
g_trainingSet = {}
g_neuralNetwork = {}
g_testSet = []

def char2vec(char):
    return [
        -1 if pixel == '.' else 1
        for line in char
        for pixel in line
    ]

def dot(x, y):
    return sum([x_i * y_i for x_i, y_i in zip(x, y)])

def initializeBiasAndWeights(trainingSet):
    b = random()
    w = []

    for _ in trainingSet.items():
        w.append(random())
    
    return {'b': b, 'w': w}

def initializeTrainingItem(trainingSet, id, imageData):
    trainingItem = {
        'image_id' : id, 
        'image' :  imageData,
        'vector_data': char2vec(imageData)
    }
    trainingSet[id] = trainingItem

def initializePerceptron(trainingSet, id):
    
    perceptronNode = []
    for key, value in trainingSet.items():
        vectorData = value['vector_data']
        index = 1 if (key == id) else 0

        perceptronNode.append([vectorData, index])

    info = initializeBiasAndWeights(trainingSet)
    b = info['b']
    w = info['w']

    return {'node': perceptronNode, 'w': w, 'b': b}

def initializeTrainingSet(trainingSet, trainingData):
    i = 0
    lengthOfTrainData = len(trainingData)
    for c in ascii_lowercase:
        if (i+1<=lengthOfTrainData):
            imageData = trainingData[i]
            initializeTrainingItem(trainingSet, c, imageData)
            i += 1
        else:
            break

def initializeNeuralNetwork(trainingSet, neuralNetwork):
    for key in trainingSet.keys():
        perceptron = initializePerceptron(trainingSet, key)
        neuralNetwork[key] = perceptron

def initializeTestSet(testSet, testData):
    for i in testData:
        testSet.append(char2vec(i))

def learn(testItem, perceptron):

    test = testItem['vector_data']
    correctAnswer = testItem['correct_answer']
    b = perceptron['b']
    w = perceptron['w']



def thresholdFromReferenceSet(referenceSetItem, id, y):
    referenceItem = referenceSetItem[id]

    return referenceItem['image_id'] if y >= 0 else NULL_ID

def predict(referenceSetItem, nnTrainResultSetItem, itemToTest):
    prediction = NULL_ID
    current = NULL_ID
    for key, nnTrainResult in nnTrainResultSetItem.items():
        w = nnTrainResult['w']
        b = nnTrainResult['b']
        response = dot(w, itemToTest) + b
        current = thresholdFromReferenceSet(referenceSetItem, key, response)
        if (current != NULL_ID):
            prediction = current
            break
    
    return prediction


initializeTrainingSet(g_trainingSet, TRAINING_DATA)

initializeTestSet(g_testSet, TEST_DATA)

initializeNeuralNetwork(g_neuralNetwork, g_trainingSet)

for x in g_testSet:
    prediction = predict(g_trainingSet, g_neuralNetwork, x)
    print(f"Prediction: {prediction}")
