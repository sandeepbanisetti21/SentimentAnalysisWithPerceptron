import calculateFscore
import sys

def main():
    iterations = 100
    print(iterations)
    vanillamean = 0
    avgmean = 0
    print('----------------------------Vanilla Modeling--------------------------------')
    for x in range(iterations):
        print("iteration : {0}".format(x))
        vanillamean += calculateFscore.run('data/train-labeled.txt', 'data/dev-text.txt', 'data/dev-key.txt', 'vanillamodel.txt')
    print("total mean value after {0} iterations is {1}".format(iterations,vanillamean/iterations))
    print('----------------------------Average Modeling--------------------------------')
    for x in range(iterations):
        print("iteration : {0}".format(x))
        avgmean += calculateFscore.run('data/train-labeled.txt', 'data/dev-text.txt', 'data/dev-key.txt', 'averagedmodel.txt')
    print("total mean value after {0} iterations is {1}".format(iterations,avgmean/iterations))

if __name__ == '__main__':
    main()