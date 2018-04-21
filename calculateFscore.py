import percepclassify3
import perceplearn3
import sys
import math

trueFake = []
posneg = []
trueFakeGold = []
posnegGold = []

TRUE = 'True'
FAKE = 'Fake'
POSITIVE = 'Pos'
NEGATIVE = 'Neg'

def main():
    perceplearn3.run(sys.argv[1])
    percepclassify3.run('vanillamodel.txt',sys.argv[2])
    processOutput()
    processGoldStandard(sys.argv[3])

    (a,b,c) = get_performance_measure(trueFake,trueFakeGold,TRUE)
    process(a,b,c,TRUE)

    (a1,b1,c1) = get_performance_measure(trueFake,trueFakeGold,TRUE)
    process(a1,b1,c1,FAKE)

    (a2,b2,c2) = get_performance_measure(posneg,posnegGold,POSITIVE)
    process(a2,b2,c2,POSITIVE)
        
    (a3,b3,c3) = get_performance_measure(posneg,posnegGold,NEGATIVE)
    process(a3,b3,c3,NEGATIVE)
    
    avgval = mean((c,c1,c2,c3))
    print('average f1 is : {0}'.format(avgval))

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def process(precission,recall, f1, cls):
    print('class {0}, precission {1},recall {2}, f1 {3}'.format(cls,precission,recall,f1))


def processOutput():
    with open('percepoutput.txt', encoding="utf8") as f:
        inputlines = f.readlines()
    for line in inputlines:
       truefakeval, posnegval = handleLine(line)
       trueFake.append(truefakeval)
       posneg.append(posnegval) 

def processGoldStandard(filename):
    with open(filename, encoding="utf8") as f:
        inputlines = f.readlines()
    for line in inputlines:
       truefakeval, posnegval = handleLine(line)
       trueFakeGold.append(truefakeval)
       posnegGold.append(posnegval) 


def handleLine(line):
    words = line.strip().split(" ")
    return words[1],words[2]


def get_performance_measure(prediction, dev_gold, cls):
    #print(prediction)
    #print(dev_gold)
    total = len(prediction)
    pos_cls_gold = len(list(filter(lambda x: x == cls, dev_gold)))
    pos_cls_pred = len(list(filter(lambda x: x == cls, prediction)))
    true_positive = 0
    for pred, gold in zip(prediction, dev_gold):
        if pred==cls and pred == gold:
            true_positive+=1
    # print(cls, true_positive, pos_cls_pred, pos_cls_gold)
    precision = true_positive/pos_cls_pred
    recall = true_positive/pos_cls_gold
    f1 = 2/((1/precision) + (1/recall))
    return precision, recall, f1

if __name__ == '__main__':
    main()