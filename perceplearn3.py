import re
import json
from pprint import pprint
from collections import defaultdict
import math
import sys
from random import shuffle
import time

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

frequency = 1
UNKNOWN = 'unknown-word-id'

stopwords = ['hers', "it's", 'too', 'who', 'he', "won't", 'then', 'up', 'between', 'ma', 'whom', 'over', 'theirs', 'on', 'just', 'isn', 'while', "don't", 'shan', "shan't", "that'll", 'their', 'does', 'yourself', 'if', 'of', 'to', 'most', 'down', 'off', "doesn't", "you'll", 'such', 's', 'will', 'under', 'its', 'mightn', 'ours', 'not', 'into', 'ourselves', 'me', 'few', 'below', 'own', 'weren', "haven't", "didn't", "aren't", 'the', 'during', 'my', 'a', 've', 'through', 'and', 'can', 'yourselves', 'ain', "wouldn't", "you'd", 'once', 'should', "wasn't", 'above', 'her', 'at', 'she', 'has', 't', 're', 'yours', 'him', "she's", 'have', 'been', 'i', 'themselves', 'so', 'again', 'll', 'with', 'himself', 'there', 'y', 'it', 'his', 'be', 'or',
             'don', 'each', 'itself', 'that', 'didn', 'until', 'from', 'won', 'being', 'how', 'you', 'now', 'other', 'is', 'some', 'are', 'same', 'very', "hasn't", 'haven', 'o', 'hadn', 'any', 'against', "couldn't", 'this', 'having', 'in', 'shouldn', 'those', 'what', 'because', 'them', 'mustn', "shouldn't", 'was', 'did', 'here', 'all', 'herself', "should've", 'd', "hadn't", "mustn't", "you've", 'doesn', "isn't", 'needn', 'our', 'further', 'were', 'why', "you're", 'nor', 'myself', 'm', 'aren', 'wasn', 'doing', 'these', "needn't", "mightn't", 'by', 'about', 'more', 'only', 'couldn', 'wouldn', 'before', 'they', "weren't", 'where', 'which', 'do', 'when', 'no', 'as', 'an', 'am', 'both', 'hasn', 'had', 'your', 'out', 'than', 'we', 'after', 'for', 'but']
#stopwords = []


def handleWordData(elements):
    wordData = defaultdict(int)
    for key in elements.keys():
        if key not in stopwords:
            wordData[key] = elements[key]
    return wordData

def getUnigramData(text,featureInfo):
    words = text.lower().split(" ")
    featureData = defaultdict(int)
    for word in words:
        featureData[word] += 1
        featureInfo[word] += 1
    return featureData,featureInfo


def getFeatures(inputData):
    featureData = []
    featureInfo = defaultdict(int)
    for line in inputData:
        data = {}
        #line = re.sub('[!.:;()\[\],\",\']', '', line)
        words = line.split(" ")
        data[ID] = words[0]
        data[AUTHENTICITY] = words[1]
        data[SENTIMENT] = words[2]
        wordData,featureInfo = getUnigramData(" ".join(words[3:]),featureInfo)
        data[FEATURES] = wordData
        #pprint(wordData)
        featureData.append(data)
    #pprint(featureInfo)    
    return featureData,featureInfo

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


def doVanillaTraining(featureData, rarewords):
    authenticity = {}
    sentiment = {}

    authenticity[WEIGHT], authenticity[BIAS] = trainVanilla(
        featureData, AUTHENTICITY, TRUE, FAKE, getDefaultTrainData(),rarewords)
    
    clonedData = clone(featureData)
    for data in clonedData:
         data[FEATURES] = handleWordData(data[FEATURES])

    sentiment[WEIGHT], sentiment[BIAS] = trainVanilla(
        clonedData, SENTIMENT, POSITVE, NEGATIVE, getDefaultTrainData(),rarewords)
    
    return authenticity, sentiment

def clone(featureData):
    from copy import deepcopy
    clonedData = deepcopy(featureData)
    return clonedData

def doAverageTraining(featureData,rarewords):
    authenticity = {}
    sentiment = {}
    authenticity[WEIGHT], authenticity[BIAS] = trainAvg(
        featureData, AUTHENTICITY, TRUE, FAKE, getDefaultTrainData(),rarewords)
    clonedData = clone(featureData)
    for data in clonedData:
         data[FEATURES] = handleWordData(data[FEATURES])    
    sentiment[WEIGHT], sentiment[BIAS] = trainAvg(
        clonedData, SENTIMENT, POSITVE, NEGATIVE, getDefaultTrainData(),rarewords)
    return authenticity, sentiment


def trainVanilla(featureData, featureType, positiveValue, negativeValue, trainData,rarewords):
    weight = trainData[WEIGHT]
    bias = trainData[BIAS]
    for x in range(ITERATIONS):
        # shuffle(featureData)
        for data in featureData:
            fired = computeActivation(data, bias, weight,rarewords)
            expected = computeExpectedNumericalValue(
                positiveValue, negativeValue, data[featureType])
            if fired*expected <= 0:
                weight, bias = updateWeightsAndBias(
                    weight, bias, expected, data[FEATURES],rarewords)
    return weight, bias


def trainAvg(featureData, featureType, positiveValue, negativeValue, trainData,rarewords):
    weight = trainData[WEIGHT]
    bias = trainData[BIAS]
    cachedWeights = defaultdict(float)
    beta = 0
    count = 1
    for x in range(ITERATIONS):
        # shuffle(featureData)
        for data in featureData:
            fired = computeActivation(data, bias, weight,rarewords)
            expected = computeExpectedNumericalValue(
                positiveValue, negativeValue, data[featureType])
            if fired*expected <= 0:
                weight, bias = updateWeightsAndBias(
                    weight, bias, expected, data[FEATURES],rarewords)
                cachedWeights, beta = updateCachedWeightsAndBias(
                    cachedWeights, beta, expected, data[FEATURES], count,rarewords)
            count += 1
    return calculateAverageWeights(weight, cachedWeights, bias, beta, count)


def computeActivation(data, bias, weight,rarewords):
    vectorData = data[FEATURES]
    a = 0
    for x in vectorData.keys():
        if x in rarewords:
            a += weight[UNKNOWN]
        else:    
            a += weight[x]*vectorData[x]
    return a+bias


def computeExpectedNumericalValue(positiveValue, negativeValue, featureValue):
    if featureValue == positiveValue:
        return 1
    else:
        return -1


def updateWeightsAndBias(weight, bias, expected, features, rarewords):
    for key in features.keys():
        if key in rarewords:
            weight[UNKNOWN] += expected*features[key]
        else:    
            weight[key] += expected*features[key]    
    bias = bias + expected
    return weight, bias


def updateCachedWeightsAndBias(cachedWeight, beta, expected, features, count,rarewords):
    for key in features.keys():
        cachedWeight[key] += expected*count*features[key]

    beta = beta + expected*count
    return cachedWeight, beta


def calculateAverageWeights(weights, cachedWeights, bias, beta, count):
    inverse = 1/count
    for key in weights.keys():
        weights[key] = weights[key] - inverse*cachedWeights[key]
    bias = bias - inverse*beta
    return weights, bias

def getRareWords(featureInfo):
    rareword = set()
    for key in featureInfo:
        if featureInfo[key] <= frequency:
            rareword.add(key)
    return rareword

def main():
    vanilla = {}
    average = {}
    filename = sys.argv[1]
    inputData = readFile(filename)
    featureData,featureInfo = getFeatures(inputData)
    rarewords = getRareWords(featureInfo)
    vanilla[AUTHENTICITY], vanilla[SENTIMENT] = doVanillaTraining(featureData,rarewords)
    average[AUTHENTICITY], average[SENTIMENT] = doAverageTraining(clone(featureData),rarewords)
    # pprint(vanilla)
    writeTofile(vanilla, 'vanillamodel.txt')
    writeTofile(average, 'averagedmodel.txt')


def run(fileName):
    start = time.clock()
    vanilla = {}
    average = {}
    inputData = readFile(fileName)
    featureData,featureInfo = getFeatures(inputData)
    rarewords = getRareWords(featureInfo)
    vanilla[AUTHENTICITY], vanilla[SENTIMENT] = doVanillaTraining(featureData,rarewords)
    average[AUTHENTICITY], average[SENTIMENT] = doAverageTraining(clone(featureData),rarewords)
    writeTofile(vanilla, 'vanillamodel.txt')
    writeTofile(average, 'averagedmodel.txt')
    print('time taken {0}'.format(time.clock() - start))


if __name__ == '__main__':

    main()
