#!/usr/bin/env python3

# Hebb Net example - Character recognition
#
# See Fausett, Example 2.8, pp. 55-56
#

referenceSet = {}
NULL_ID = ' ' 
nnTrainResultSet = {}

def addToReferenceSet(referenceSetItem, id, imageItem):
    referenceItem = {
        'image_id' : id, 
        'image' :  imageItem
    }
    referenceSetItem[id] = referenceItem

def char2vec(char):
    return [
        -1 if pixel == '.' else 1
        for line in char
        for pixel in line
    ]

def dot(x, y):
    return sum([x_i * y_i for x_i, y_i in zip(x, y)])

def train(referenceSetItem, id):
    
    trainingSet = []
    for key, value in referenceSetItem.items():
        image = value['image']

        index = -1
        if (key == id):
            index = +1
        else:
            index = -1
        trainingSet.append([char2vec(image), index])

    b = 0
    w = [0] * len(trainingSet[0][0])

    for x, y in trainingSet:
        for i in range(len(w)):
            w[i] = w[i] + x[i] * y
            b = b + y
    
    return {'training_set': trainingSet, 'w': w, 'b': b}

def thresholdFromReferenceSet(referenceSetItem, id, y):
    referenceItem = referenceSetItem[id]

    return referenceItem['image_id'] if y >= 0 else NULL_ID

def trainAll(referenceSetItem, nnTrainResultSetItem):
    for key in referenceSetItem.keys():
        nnTrainResult = train(referenceSetItem, key)
        nnTrainResultSetItem[key] = nnTrainResult

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

#Training
addToReferenceSet(referenceSet, 'X', 
[
    '#...#',
    '.#.#.',
    '..#..',
    '.#.#.',
    '#...#',
])

addToReferenceSet(referenceSet, 'O',  
[
    '.###.',
    '#...#',
    '#...#',
    '#...#',
    '.###.',
])
#Training

#Testing
TEST_O = [
    '..##.',
    '.#..#',
    '.#..#',
    '.#..#',
    '..##.',
]

TEST_X = [
    '#....',
    '.#.#.',
    '..#..',
    '.#.#.',
    '....#',
]

TEST_SET = [
    char2vec(TEST_O),
    char2vec(TEST_X),
]

#Testing

trainAll(referenceSet, nnTrainResultSet)

for x in TEST_SET:
    prediction = predict(referenceSet, nnTrainResultSet, x)
    print(f'Prediction: {prediction}')
