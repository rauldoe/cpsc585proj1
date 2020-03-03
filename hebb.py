#!/usr/bin/env python3

# Hebb Net example - Character recognition
#
# See Fausett, Example 2.8, pp. 55-56
#

TRAINING_X = [
    '#...#',
    '.#.#.',
    '..#..',
    '.#.#.',
    '#...#',
]

TRAINING_O = [
    '.###.',
    '#...#',
    '#...#',
    '#...#',
    '.###.',
]


def char2vec(char):
    return [
        -1 if pixel == '.' else 1
        for line in char
        for pixel in line
    ]


TRAINING_SET = [
    [char2vec(TRAINING_X), +1],
    [char2vec(TRAINING_O), -1]
]

b = 0
w = [0] * len(TRAINING_SET[0][0])

for x, y in TRAINING_SET:
    for i in range(len(w)):
        w[i] = w[i] + x[i] * y
        b = b + y


def dot(x, y):
    return sum([x_i * y_i for x_i, y_i in zip(x, y)])


def threshold(y):
    return 'X' if y >= 0 else 'O'


for x, y in TRAINING_SET:
    response = dot(w, x) + b
    print(f'Prediction: {threshold(response)}, Label: {threshold(y)}')

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

for x in TEST_SET:
    response = dot(w, x) + b
    print(f'Prediction: {threshold(response)}')

    #this is a test
