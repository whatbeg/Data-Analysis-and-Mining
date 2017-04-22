import random
import math
import matplotlib.pyplot as pyplot
import numpy as np


def initNetwork(layer):
    weights = np.zeros((2, 5))
    np.random.seed(10)
    weights[0] = np.random.normal(0, 1**-0.5, 5)
    weights[1] = np.random.normal(0, 2**-0.5, 5)
    print(weights)
    gradients = np.array([
        [],
        []
    ])
    return weights, gradients


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigd(x):
    return x * (1.0-x)


def updateOutput(input, weights):

    output = np.zeros((3, 5))    # output = [[input], [hidden], [output]]
    output[0].fill(input)
    # print(sigmoid(output[0] * weights[0]))
    output[1, :] = map(sigmoid, list(output[0]*weights[0]))
    output[2] = output[1] * weights[1]

    y = np.sum(output[2])
    return output, y


def updategradient(input, y, output, weights, target, lr):

    deltaW2 = (y - target) * output[1]
    deltaW1 = np.zeros(len(deltaW2))
    for _ in range(len(deltaW2)):
        deltaW1[_] = (y - target) * weights[1][_] * sigd(output[1][_]) * input

    weights[0] = weights[0] - lr * deltaW1
    weights[1] = weights[1] - lr * deltaW2

    return weights


def randomgenerate(num):

    x = []
    y = []
    np.random.seed(30)
    for _ in range(num):
        xi = np.random.uniform(0, 2*3.142)
        yi = math.sin(xi)
        x.append(xi)
        y.append(yi)

    return x, y


def train(trainSize):

    x, y = randomgenerate(trainSize)
    pyplot.scatter(x, y)
    pyplot.xlabel("train_x")
    pyplot.ylabel("train_y")
    pyplot.show()
    weights, gradients = initNetwork(2)
    learningRate = 0.03
    MaxEpoch = 200
    epoch = 0
    Loss = []
    while epoch < MaxEpoch:
        epoch += 1
        Error = 0
        idx = range(trainSize)
        np.random.shuffle(idx)
        for i in idx:
            xi, yi = x[i], y[i]
            output, yout = updateOutput(xi, weights)
            Error += 0.5 * (yout - yi) ** 2
            weights = updategradient(xi, yout, output, weights, yi, learningRate)
        print "[Epoch {}] Average Loss is {}".format(epoch, Error / trainSize)
        Loss.append(Error / trainSize)
    pyplot.title("Loss")
    pyplot.plot(Loss)
    pyplot.show()
    return weights

if __name__ == '__main__':

    model = train(600)
    print(model)
    testx, testy = randomgenerate(100)
    OUT = []
    for tx in testx:
        _, out = updateOutput(tx, model)
        OUT.append(out)
    pyplot.title("Test")
    pyplot.xlabel("test_x")
    pyplot.ylabel("test_y")
    pyplot.scatter(testx, OUT, label="model output", s=15)
    pyplot.scatter(testx, testy, label="test set", s=15)
    pyplot.legend(loc='upper right')
    pyplot.show()
