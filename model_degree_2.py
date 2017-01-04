import math
import random

def curve(X, a, b, c):
    return [a * x ** 2 + b * x + c for x in X]

def sse(Y, Y_pred):
    return sum([(y - y_pred) ** 2 for y, y_pred in zip(Y, Y_pred)])

def loss(Y, a, b, c, X):
    return sse(Y, curve(X, a, b, c))

def avg_loss(Y, a, b, c, X):
    return math.sqrt(loss(Y, a, b, c, X) / len(X))

def average(X):
    return sum(X) / len(X)

def update(Y, a, b, c, X, learning_rate):
    Y_pred = curve(X, a, b, c)
    dydc = [2 * (y_pred - y) for y_pred, y in zip(Y_pred, Y)]
    dydb = [x * d for x, d in zip(X, dydc)]
    dyda = [x * d for x, d in zip(X, dydb)]
    a -= learning_rate * average(dyda)
    b -= learning_rate * average(dydb)
    c -= learning_rate * average(dydc)
    return a, b, c

if __name__ == '__main__':
    n_samples = 100
    a, b, c = 7, 3, 22
    learning_rate = 0.5
    X = [random.random() for _ in range(n_samples)]
    Y = curve(X, a, b, c)
    print(X)
    print(Y)
    a_guess, b_guess, c_guess = -1., 1., -1
    print('initial values: a = %d, b = %d, c = %d' % (a_guess, b_guess, c_guess))
    for i in range(3000):
        a_guess, b_guess, c_guess = update(Y, a_guess, b_guess, c_guess, X, learning_rate)
        print('step %d : a_guess = %.2f, b_guess = %.2f, c_guess = %.2f. Average loss : %.2f' % (i, a_guess, b_guess, c_guess, avg_loss(Y, a_guess, b_guess, c_guess, X)))
    print(avg_loss(Y, a_guess, b_guess, c_guess, X))
    
