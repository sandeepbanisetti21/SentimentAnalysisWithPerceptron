import re
import json
import sys

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
STOPWORDS = 'stopwords'
UNKNOWN = 'UNKNOWN'

def handleStopwords(elements,stopwords):
    for element in elements:
        if element in stopwords:
            elements.remove(element)
    return elements

def extractFeatures(filename):
    with open(filename, encoding="utf8") as f:
        inputlines = f.readlines()
    featureData = []
    for line in inputlines:
        data = {}
        line = re.sub('[!.:;()\[\],\",\']', '', line)
        words = line.strip().split(" ")
        data[ID] = words[0]
        data[FEATURES] = words[1:]
        featureData.append(data)    
    return featureData

def classify(featureData,trainingData,positiveClass, negativeClass, featuretype):
    output = {}
    bias = trainingData[BIAS]
    weight = trainingData[WEIGHT]
    featureInfo = weight.keys()
    for test in featureData:
        fired = computeActivation(test, bias, weight,featureInfo)
        output[test[ID]] = getClassInfo(fired,positiveClass,negativeClass, featuretype)
    return output

def getClassInfo(fired, positiveClass, negativeClass,featureType):
    if featureType == AUTHENTICITY:
        if fired > 0:
            return TRUE
        else:
            return FAKE
    else:
        if fired > 0:
            return POSITVE
        else:
            return NEGATIVE            

def computeActivation(data, bias, weight,featureInfo):
    vectorData = data[FEATURES]
    a = 0
    for x in vectorData:
        if x in featureInfo:
            a += weight[x]
        else:
            a += weight[UNKNOWN]    
    return a+bias 


def readData(fileName):
    data = json.load(open(fileName, encoding='utf-8'))
    return data

def extractTrainingData(data,featureType):
    trainingData = {}
    trainingData[WEIGHT] = data[featureType][WEIGHT]
    trainingData[BIAS] =data[featureType][BIAS]
    return trainingData     

def writeTofile(authenticity, sentiment, featureData):
    output = []
    for x in featureData:
        id = x[ID]
        ans = id+' '+authenticity[id]+' '+sentiment[id]
        output.append(ans)
    file = open('percepoutput.txt', 'w', encoding='utf-8')
    for item in output:
      file.write("%s\n" % item)

def main():
    textData = readData(sys.argv[1])  
    authenticityData = extractTrainingData(textData,AUTHENTICITY)
    sentimentData = extractTrainingData(textData,SENTIMENT)
    featureData = extractFeatures(sys.argv[2])
    authenticity =  classify(featureData, authenticityData, TRUE, FAKE , AUTHENTICITY)
    sentiment =  classify(featureData, sentimentData, POSITVE, NEGATIVE ,SENTIMENT)
    writeTofile(authenticity,sentiment,featureData)

def run(modelName,testFileName):
    textData = readData(modelName)
    
    authenticityData = extractTrainingData(textData,AUTHENTICITY)
    sentimentData = extractTrainingData(textData,SENTIMENT)    
    featureData = extractFeatures(testFileName)
    authenticity =  classify(featureData, authenticityData, TRUE, FAKE , AUTHENTICITY)
    sentiment =  classify(featureData, sentimentData, POSITVE, NEGATIVE ,SENTIMENT)
    writeTofile(authenticity,sentiment,featureData)


        
if __name__ == '__main__':
    main()