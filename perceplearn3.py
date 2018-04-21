import re
import json
from pprint import pprint
from collections import defaultdict
import math
import sys

#DATA
TRUE = 'True'
FAKE = 'Fake'
POSITVE = 'Pos'
NEGATIVE = 'Neg'

#FEATURE TYPES
AUTHENTICITY = 'Authenticity'
SENTIMENT = 'Sentiment'
ID = 'id'
FEATURES = 'features'

#TrainData
WEIGHT = 'Weight'
BIAS = 'Bias'
ITERATIONS = 100



def handleStopWords():
    return []


def getFeatures(inputData):
    featureData = []
    featureInfo = []
    for line in inputData:
        data = {}
        line = re.sub('[!.:;()\[\],\",\']', '', line)
        words = line.strip().split(" ")
        data[ID] = words[0]
        data[AUTHENTICITY] = words[1]
        data[SENTIMENT] = words[2]
        data[FEATURES] = words[3:]
        print(data)
        featureData.append(data)
        featureInfo.extend(words[3:])
    return featureData


def readFile(filename):
    with open(filename, encoding='utf8') as f:
        inputlines = f.readlines()
    return inputlines


def main():
    writingData = {}
    filename = sys.argv[1]
    inputData = readFile(filename)
    featureData = getFeatures(inputData)
    handleStopWords()
    vanillaModel = doVanillaTraining(featureData)
    

    writeTofile(vanillaModel,'vanillamodel.txt')    


def doVanillaTraining(featureData):
    vanillaModel ={}
    vanillaModel[AUTHENTICITY][WEIGHT],vanillaModel[AUTHENTICITY][BIAS] = trainVanilla(
        featureData, AUTHENTICITY, TRUE, FAKE, getDefaultTrainData())

    vanillaModel[SENTIMENT][WEIGHT], vanillaModel[SENTIMENT][BIAS] = trainVanilla(
        featureData, SENTIMENT, POSITVE, NEGATIVE, getDefaultTrainData())
    return vanillaModel

def writeTofile(data, name):
    x = json.dumps(data, ensure_ascii=False)
    with open(name, 'w', encoding='utf-8') as file:
        file.write(x)

def getDefaultTrainData():
    trainData = {}
    trainData[WEIGHT] = defaultdict(float)
    trainData[BIAS] = 0
    return trainData


def trainVanilla(featureData, featureType, positiveValue, negativeValue, trainData):
    weight = trainData['weight']
    bias = trainData['bias']
    for x in range(ITERATIONS):
        for data in featureData:
            fired = computeActivation(data, bias, weight)
            expected = computeExpectedNumericalValue(
                positiveValue, negativeValue, data[featureType])
            if fired*expected <= 0:
                weight, bias = updateWeightsAndBias(
                    weight, bias, expected, data[FEATURES])
    return weight, bias


def computeActivation(data, bias, weight):
    vectorData = data[FEATURES]
    a = 0
    for x in vectorData:
        a += weight[x]
    return a+bias


def computeExpectedNumericalValue(positiveValue, negativeValue, featureValue):
    if featureValue == positiveValue:
        return 1
    else:
        return -1


def updateWeightsAndBias(weight, bias, expected, features):
    for feature in features:
        weight[feature] += expected

    bias = bias + expected
    return weight, bias


if __name__ == '__main__':
    main()
