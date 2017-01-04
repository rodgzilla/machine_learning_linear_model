import math
import random

def line(X, a, b):
    return [a*x + b for x in X]

def sse(Y, Y_pred):
    return sum([(y - y_pred) ** 2 for y, y_pred in zip(Y, Y_pred)])

def loss(Y, a, b, X):
    return sse(Y, line(X, a, b))

def avg_loss(Y, a, b, X):
    return math.sqrt(loss(Y, a, b, X) / len(X))

def average(X):
    return sum(X) / len(X)

def update(Y, a, b, X, learning_rate):
    Y_pred = line(X, a, b)
    # print(a, b)
    # print('(X, ypred)', list(zip(X, Y_pred)))
    dydb = [2 * (y_pred - y) for y_pred, y in zip(Y_pred, Y)]
    dyda = [x * d for x, d in zip(X, dydb)]
    a -= learning_rate * average(dyda)
    b -= learning_rate * average(dydb)
    return a, b

if __name__ == '__main__':
    n_samples = 30
    a, b = 2, 2
    learning_rate = 0.1
    X = [random.random() for _ in range(n_samples)]
    Y = line(X, a, b)
    print(X)
    print(Y)
    a_guess, b_guess = -1., 1.
    print('initial values: a = %d, b = %d' % (a_guess, b_guess))
    for i in range(400):
        a_guess, b_guess = update(Y, a_guess, b_guess, X, learning_rate)
        print('step %d : a_guess = %.2f, b_guess = %.2f. Average loss : %.2f' % (i, a_guess, b_guess, avg_loss(Y, a_guess, b_guess, X)))
    print(avg_loss(Y, a_guess, b_guess, X))
    
