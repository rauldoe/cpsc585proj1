#Source https://www.youtube.com/watch?v=OVHc-7GYRo4
#Title: Single-Layer Perceptron: Background & Python Code

import sys
import random

#Use the provided datasets from dataset.py
from dataset import *


#Convert the character to a vector
def char2vec(char):
    return [
        -1 if pixel == '.' else 1
        for line in char
        for pixel in line
    ]

#Predict what the letter is
def predict(inputs, weights):
    threshold=0.0
    total_activation=0.0
    for input, weight in zip(inputs,weights):
        total_activation += input*weight
    #print(total_activation)
    return 1.0 if total_activation >= threshold else 0.0

#Calculate prediction accuracy, provided inputs and associated weights
def accuracy(matrix, weights):
    num_correct = 0.0
    preds = []
    for i in range(len(matrix)):
        pred = predict(matrix[i][:-1], weights)
        preds.append(pred)
        if pred==matrix[i][-1]:
            num_correct += 1.0
    print("\nPredictions:", preds)
    return num_correct/float(len(matrix))

#Train weights
def train_weights(matrix, weights, nb_epoch=10, l_rate=1.0):
    for epoch in range(nb_epoch):
        cur_acc= accuracy(matrix, weights)
        print("Epoch %d"%epoch)
        #print("\nEpoch %d \nWeights: "%epoch,weights)
        print("Accuracy: ",cur_acc)

        if(cur_acc==1.0): break

        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1],weights)

            #This is (desired - perceptron output). The desired result is the last element of the training set. We appended the correct output (1 or 0) to each letter.
            error = matrix[i][-1]-prediction 

            for j in range(len(weights)):
                weights[j]=weights[j] + (l_rate*error * matrix[i][j])
            
            
    return weights


def main():

    #Train Network
    data = []
    SavedWeights = []
    for i in range(0,26):
        data = []
        print("\n\n------------------------------")
        print(chr(65 + i))
        for j in range(0,26):
            if(i != j):
                temparray = char2vec(TRAINING_DATA[j])
                temparray.append(0)
                data.append(temparray)
            else:
                temparray = char2vec(TRAINING_DATA[j])
                temparray.append(1)
                data.append(temparray)
        weights = []
        for i in range(len(char2vec(TRAINING_DATA[0]))):
            weights.append(random.random())
        SavedWeights.append(train_weights(data, weights=weights, nb_epoch=10000, l_rate=1))

    #Check using test data set
    #There are 26 different sets of weights for 26 different perceptrons for Letter/Not That Letter
    testData = []
    predictionArray = []
    for i in range(0,26):
        temparray = char2vec(TEST_DATA[i])
        testData.append(temparray)
    for i in range(0,26):
        predictionArray.append(predict(testData[i],SavedWeights[i]))

    #Print out final predictions.
    print("\n------------------------------")
    print("Prediction on test data:")
    print(predictionArray)
    correctPredictions = 0.0
    for i in range(0,26):
        if(predictionArray[i] == 1.0):
            correctPredictions += 1
    correctPercentage = (correctPredictions/26.0) * 100

    #Amount of perceptrons that correct predict that the input letter is in fact that letter. Example: Is an A is an A?; Yes/No
    print("Correct percentage: " + str(correctPercentage))

    #TODO
    #For every epoch of every pereptron training, check the accuracy of it to correctly predict 'letter/not letter' and graph it.
    
if __name__ == '__main__':
    main()