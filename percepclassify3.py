import re
import json
import sys
from collections import defaultdict

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

UNKNOWN = 'unknown-word-id'

stopwords = ['hers', "it's", 'too', 'who', 'he', "won't", 'then', 'up', 'between', 'ma', 'whom', 'over', 'theirs', 'on', 'just', 'isn', 'while', "don't", 'shan', "shan't", "that'll", 'their', 'does', 'yourself', 'if', 'of', 'to', 'most', 'down', 'off', "doesn't", "you'll", 'such', 's', 'will', 'under', 'its', 'mightn', 'ours', 'not', 'into', 'ourselves', 'me', 'few', 'below', 'own', 'weren', "haven't", "didn't", "aren't", 'the', 'during', 'my', 'a', 've', 'through', 'and', 'can', 'yourselves', 'ain', "wouldn't", "you'd", 'once', 'should', "wasn't", 'above', 'her', 'at', 'she', 'has', 't', 're', 'yours', 'him', "she's", 'have', 'been', 'i', 'themselves', 'so', 'again', 'll', 'with', 'himself', 'there', 'y', 'it', 'his', 'be', 'or',
             'don', 'each', 'itself', 'that', 'didn', 'until', 'from', 'won', 'being', 'how', 'you', 'now', 'other', 'is', 'some', 'are', 'same', 'very', "hasn't", 'haven', 'o', 'hadn', 'any', 'against', "couldn't", 'this', 'having', 'in', 'shouldn', 'those', 'what', 'because', 'them', 'mustn', "shouldn't", 'was', 'did', 'here', 'all', 'herself', "should've", 'd', "hadn't", "mustn't", "you've", 'doesn', "isn't", 'needn', 'our', 'further', 'were', 'why', "you're", 'nor', 'myself', 'm', 'aren', 'wasn', 'doing', 'these', "needn't", "mightn't", 'by', 'about', 'more', 'only', 'couldn', 'wouldn', 'before', 'they', "weren't", 'where', 'which', 'do', 'when', 'no', 'as', 'an', 'am', 'both', 'hasn', 'had', 'your', 'out', 'than', 'we', 'after', 'for', 'but']

def getUnigramData(text):
    words = text.lower().split(" ")
    featureData = defaultdict(int)
    for word in words:
        featureData[word] += 1
    return featureData


def extractFeatures(filename):
    with open(filename, encoding="utf8") as f:
        inputlines = f.readlines()
    featureData = []
    for line in inputlines:
        data = {}
        #line = re.sub('[!.:;()\[\],\",\']', '', line)
        words = line.split(" ")
        data[ID] = words[0]
        #data[FEATURES] = " ".join(words[1:]).lower().split(" ")
        data[FEATURES] = getUnigramData(" ".join(words[1:]))
        featureData.append(data)    
    return featureData

def classify(featureData,trainingData,positiveClass, negativeClass, featuretype,stopwords):
    output = {}
    bias = trainingData[BIAS]
    weight = trainingData[WEIGHT]
    for test in featureData:
        fired = computeActivation(test, bias, weight,stopwords)
        output[test[ID]] = getClassInfo(fired,positiveClass,negativeClass,featuretype)
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

def computeActivation(data, bias, weight,stopwords):
    vectorData = data[FEATURES]
    feautreInfo = set(weight.keys())
    a = 0
    for x in vectorData.keys():
        if x not in stopwords:
            if x not in feautreInfo:
                a += weight[UNKNOWN]*vectorData[x]
            else:    
                a += weight[x]*vectorData[x]
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
    authenticity =  classify(featureData, authenticityData, TRUE, FAKE , AUTHENTICITY,[])
    sentiment =  classify(featureData, sentimentData, POSITVE, NEGATIVE ,SENTIMENT,stopwords)
    writeTofile(authenticity,sentiment,featureData)

def run(modelName,testFileName):
    textData = readData(modelName)
    authenticityData = extractTrainingData(textData,AUTHENTICITY)
    sentimentData = extractTrainingData(textData,SENTIMENT)    
    featureData = extractFeatures(testFileName)
    authenticity =  classify(featureData, authenticityData, TRUE, FAKE , AUTHENTICITY,[])
    sentiment =  classify(featureData, sentimentData, POSITVE, NEGATIVE ,SENTIMENT,stopwords)
    writeTofile(authenticity,sentiment,featureData)


        
if __name__ == '__main__':
    main()