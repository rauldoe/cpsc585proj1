#Import random for use when initializing the weights.
import random

#Use the provided datasets from dataset.py.
from dataset import *

#Import matplotlib for plotting.
import matplotlib.pyplot as plot

#Import numpy for the dot function.
import numpy as np

#Convert the character to a vector. Function taken from hebb example.
def char2vec(char):
    return [
        -1 if pixel == '.' else 1
        for line in char
        for pixel in line
    ]

#Predict whether the letter is the desired letter or not.
def predict(inputs, weights):
    response = sum([x_i * y_i for x_i, y_i in zip(inputs, weights)])
    return 1.0 if response >= 0 else 0.0

#Calculate the error rate for training.
def errorRateTraining(trainingData, weights):
    numCorrect = 0.0
    for i in range(len(trainingData)):
        prediction = predict(trainingData[i][:-1], weights)
        if prediction == trainingData[i][-1]:
            numCorrect += 1.0
    return (1 - numCorrect/float(len(trainingData))) * 100

#Calculate how correct the particular perceptron is based on predictions from test data.
def errorRateTesting(predictions, letter):
    value = 0.0
    for k in range(0,26):
        value += predictions[k]
    if(predictions[letter] == 1.0):
        value -= 1
    if(predictions[letter] == 0):
        value += 1
    return (100 * (value/len(predictions)))

#Plot the perceptron error rate vs number of epochs.
def plotPerceptron(epochValues, errorRateValues, letter):
    plot.plot(epochValues, errorRateValues, color='blue', marker='x')
    plot.title('Error Rate versus Epoch for Perceptron ' + letter)
    plot.xlabel('Epoch')
    plot.ylabel('Error Rate %')
    plot.grid(True)
    plot.show()

#Update the weights
def updateWeights(trainingData, weights, epochs, learningRate, letter):
    epochValues = []
    errorRateValues = []
    weightsArray = []

    for epoch in range(epochs):
        errorPercentage = errorRateTraining(trainingData, weights)
        #Update the weights.
        epochValues.append(epoch)
        errorRateValues.append(errorPercentage)
        weightsArray.append(weights[:])
        if(errorPercentage <= 0): 
            return [epochValues, errorRateValues, weightsArray]
        for i in range(len(trainingData)):
            prediction = predict(trainingData[i][:-1], weights)     
            for j in range(len(weights)):
                # https://en.wikipedia.org/wiki/Perceptron - w = w + learningRate dot (desired output - perceptron output)x,ij
                weights[j] = weights[j] + np.dot(learningRate, (trainingData[i][-1] - prediction) * trainingData[i][j])
    return [epochValues, errorRateValues, weightsArray]
    


def main():

    #Trainining data.
    trainingData = []

    #Stored weights of each perceptron.
    SavedWeights = []

    #There are 26 different perceptrons. One for each letter - Letter/Not Letter.
    for i in range(0,26):
        trainingData = []
        print("\n\n------------------------------")
        letter = str(chr(65 + i))
        print(letter)
        for j in range(0,26):
            if(i != j):
                temparray = char2vec(TRAINING_DATA[j])
                temparray.append(0)
                trainingData.append(temparray)
            else:
                temparray = char2vec(TRAINING_DATA[j])
                temparray.append(1)
                trainingData.append(temparray)
        weights = []
        for i in range(len(char2vec(TRAINING_DATA[0]))):
            weights.append(random.random())

        #Update the weights. 
        returnedArrays = updateWeights(trainingData, weights, 10000, .01, letter) #[epochValues, errorRateValues, weightsArray]
        SavedWeights.append(returnedArrays[2][-1])

        #For each epoch in the training, print out the epoch, error rates and the predictions.
        for epoch in range(len(returnedArrays[0])):
            print("Epoch: " + str(epoch))           
            predictions = []
            for i in range(0,26):
                prediction = predict(trainingData[i][:-1], returnedArrays[2][epoch])
                predictions.append(prediction)
            print("Predictions: " + str(predictions))
            print("Error Percentage: " + str(returnedArrays[1][epoch]))

        #Plot the perceptron training Error Rate vs Epoch graph.
        plotPerceptron(returnedArrays[0], returnedArrays[1], letter)

    #Check using test data set.
    #There are 26 different sets of weights for 26 different perceptrons for Letter/Not That Letter.
    testData = []
    predictionArray = []
    for i in range(0,26):
        temparray = char2vec(TEST_DATA[i])
        testData.append(temparray)
    for i in range(0,26):
        predictionArray = []
        for j in range(0,26):
            predictionArray.append(predict(testData[j],SavedWeights[i]))
        print("\n------------------------------")
        print("Prediction on test data for perceptron " + str(chr(65 + i)) + ":")
        print(str(predictionArray))

        #Determine the error rate.
        testingErrorRate = errorRateTesting(predictionArray, i)
        print("Error Rate: " + str(testingErrorRate))
    
if __name__ == '__main__':
    main()