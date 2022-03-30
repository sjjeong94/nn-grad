import numpy as np
import matplotlib.pyplot as plt

import nn


def get_accuracy(model, x, y):
    x = model.forward(x)
    y_pred = np.argmax(x, axis=1)
    acc = (y_pred == y).mean()
    return acc


def test_grad():
    np.random.seed(1234)

    train_image = np.load('./data/train_image.npy')
    train_label = np.load('./data/train_label.npy')
    test_image = np.load('./data/test_image.npy')
    test_label = np.load('./data/test_label.npy')

    x_train = train_image.reshape(-1, 784) / 255.
    y_train = train_label
    x_test = test_image.reshape(-1, 784) / 255.
    y_test = test_label

    y_onehot = np.zeros((len(y_train), 10), float)
    y_onehot[np.arange(len(y_train)), y_train] = 1

    criterion = nn.CrossEntropyLoss()

    net = nn.Net([
        nn.Linear(784, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ])

    for i in range(100):
        z = net.forward(x_train)
        loss = criterion.forward(z, y_onehot)
        dloss = criterion.backward()
        d = net.backward(dloss)

        net.update(0.1)

        acc_train = get_accuracy(net, x_train, y_train) * 100
        acc_test = get_accuracy(net, x_test, y_test) * 100

        print('Epoch %4d -> loss %7.3f train %7.3f / test %7.3f' %
              (i, loss, acc_train, acc_test))


if __name__ == '__main__':
    test_grad()
