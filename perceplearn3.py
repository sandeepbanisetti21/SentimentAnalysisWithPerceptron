import re
import json
from pprint import pprint
from collections import defaultdict
import math
import sys

# DATA
TRUE = 'True'
FAKE = 'Fake'
POSITVE = 'Pos'
NEGATIVE = 'Neg'

# FEATURE TYPES
AUTHENTICITY = 'Authenticity'
SENTIMENT = 'Sentiment'
ID = 'id'
FEATURES = 'features'
STOPWORDS = 'stopwords'

# TrainData
WEIGHT = 'Weight'
BIAS = 'Bias'
ITERATIONS = 16

stopwords = ['hers', "it's", 'too', 'who', 'he', "won't", 'then', 'up', 'between', 'ma', 'whom', 'over', 'theirs', 'on', 'just', 'isn', 'while', "don't", 'shan', "shan't", "that'll", 'their', 'does', 'yourself', 'if', 'of', 'to', 'most', 'down', 'off', "doesn't", "you'll", 'such', 's', 'will', 'under', 'its', 'mightn', 'ours', 'not', 'into', 'ourselves', 'me', 'few', 'below', 'own', 'weren', "haven't", "didn't", "aren't", 'the', 'during', 'my', 'a', 've', 'through', 'and', 'can', 'yourselves', 'ain', "wouldn't", "you'd", 'once', 'should', "wasn't", 'above', 'her', 'at', 'she', 'has', 't', 're', 'yours', 'him', "she's", 'have', 'been', 'i', 'themselves', 'so', 'again', 'll', 'with', 'himself', 'there', 'y', 'it', 'his', 'be', 'or',
             'don', 'each', 'itself', 'that', 'didn', 'until', 'from', 'won', 'being', 'how', 'you', 'now', 'other', 'is', 'some', 'are', 'same', 'very', "hasn't", 'haven', 'o', 'hadn', 'any', 'against', "couldn't", 'this', 'having', 'in', 'shouldn', 'those', 'what', 'because', 'them', 'mustn', "shouldn't", 'was', 'did', 'here', 'all', 'herself', "should've", 'd', "hadn't", "mustn't", "you've", 'doesn', "isn't", 'needn', 'our', 'further', 'were', 'why', "you're", 'nor', 'myself', 'm', 'aren', 'wasn', 'doing', 'these', "needn't", "mightn't", 'by', 'about', 'more', 'only', 'couldn', 'wouldn', 'before', 'they', "weren't", 'where', 'which', 'do', 'when', 'no', 'as', 'an', 'am', 'both', 'hasn', 'had', 'your', 'out', 'than', 'we', 'after', 'for', 'but']


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
        # pprint(data)
        featureData.append(data)
        featureInfo.extend(words[3:])
    return featureData


def readFile(filename):
    with open(filename, encoding='utf8') as f:
        inputlines = f.readlines()
    return inputlines


def writeTofile(data, name):
    x = json.dumps(data, ensure_ascii=False)
    with open(name, 'w', encoding='utf-8') as file:
        file.write(x)


def getDefaultTrainData():
    trainData = {}
    trainData[WEIGHT] = defaultdict(float)
    trainData[BIAS] = 0
    return trainData


def doVanillaTraining(featureData):
    authenticity = {}
    sentiment = {}
    authenticity[WEIGHT], authenticity[BIAS] = trainVanilla(
        featureData, AUTHENTICITY, TRUE, FAKE, getDefaultTrainData())
    sentiment[WEIGHT], sentiment[BIAS] = trainVanilla(
        featureData, SENTIMENT, POSITVE, NEGATIVE, getDefaultTrainData())
    return authenticity, sentiment


def doAverageTraining(featureData):
    authenticity = {}
    sentiment = {}
    authenticity[WEIGHT], authenticity[BIAS] = trainAvg(
        featureData, AUTHENTICITY, TRUE, FAKE, getDefaultTrainData())
    sentiment[WEIGHT], sentiment[BIAS] = trainAvg(
        featureData, SENTIMENT, POSITVE, NEGATIVE, getDefaultTrainData())
    return authenticity, sentiment


def trainVanilla(featureData, featureType, positiveValue, negativeValue, trainData):
    weight = trainData[WEIGHT]
    bias = trainData[BIAS]
    for x in range(ITERATIONS):
        for data in featureData:
            fired = computeActivation(data, bias, weight)
            expected = computeExpectedNumericalValue(
                positiveValue, negativeValue, data[featureType])
            if fired*expected <= 0:
                weight, bias = updateWeightsAndBias(
                    weight, bias, expected, data[FEATURES])
    return weight, bias


def trainAvg(featureData, featureType, positiveValue, negativeValue, trainData):
    weight = trainData[WEIGHT]
    bias = trainData[BIAS]
    cachedWeights = defaultdict(float)
    beta = 0
    count = 1
    for x in range(ITERATIONS):
        for data in featureData:
            fired = computeActivation(data, bias, weight)
            expected = computeExpectedNumericalValue(
                positiveValue, negativeValue, data[featureType])
            if fired*expected <= 0:
                weight, bias = updateWeightsAndBias(
                    weight, bias, expected, data[FEATURES])
                cachedWeights, beta = updateCachedWeightsAndBias(
                    cachedWeights, beta, expected, data[FEATURES], count)
            count += 1
    return calculateAverageWeights(weight, cachedWeights, bias, beta, count)


def computeActivation(data, bias, weight):
    vectorData = data[FEATURES]
    a = 0
    for x in vectorData:
        #if x not in stopwords:
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


def updateCachedWeightsAndBias(cachedWeight, beta, expected, features, count):
    for feature in features:
        cachedWeight[feature] += expected*count

    beta = beta + expected*count
    return cachedWeight, beta


def calculateAverageWeights(weights, cachedWeights, bias, beta, count):
    inverse = 1/count
    for key in weights.keys():
        weights[key] = weights[key] - inverse*cachedWeights[key]
    bias = bias - inverse*beta
    return weights, bias


def main():
    vanilla = {}
    average = {}
    filename = sys.argv[1]
    inputData = readFile(filename)
    featureData = getFeatures(inputData)
    handleStopWords()
    vanilla[AUTHENTICITY], vanilla[SENTIMENT] = doVanillaTraining(featureData)
    vanilla[STOPWORDS] = stopwords
    average[AUTHENTICITY], average[SENTIMENT] = doAverageTraining(featureData)
    average[STOPWORDS] = stopwords
    # pprint(vanilla)
    writeTofile(vanilla, 'vanillamodel.txt')
    writeTofile(average, 'averagedmodel.txt')


def run(fileName):
    vanilla = {}
    average = {}
    inputData = readFile(fileName)
    featureData = getFeatures(inputData)
    handleStopWords()
    vanilla[AUTHENTICITY], vanilla[SENTIMENT] = doVanillaTraining(featureData)
    average[AUTHENTICITY], average[SENTIMENT] = doAverageTraining(featureData)
    #vanilla[STOPWORDS] = stopwords
    #average[STOPWORDS] = stopwords
    # pprint(vanilla)
    writeTofile(vanilla, 'vanillamodel.txt')
    writeTofile(average, 'averagedmodel.txt')


if __name__ == '__main__':
    main()
