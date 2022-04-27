import time
import numpy as np
import cupy as cp

import nn
import nncu
import mnist


def get_accuracy(model, x, y):
    x = model.forward(x)
    y_pred = np.argmax(x, axis=1)
    acc = (y_pred == y).mean()
    return acc


def get_accuracy_cupy(model, x, y):
    x = model.forward(x)
    y_pred = cp.argmax(x, axis=1)
    acc = (y_pred == y).mean()
    return acc


def test_grad():
    np.random.seed(1234)

    train_image, train_label, test_image, test_label = mnist.load()

    x_train = train_image / 255.
    y_train = train_label
    x_test = test_image / 255.
    y_test = test_label

    y_onehot = np.zeros((len(y_train), 10), float)
    y_onehot[np.arange(len(y_train)), y_train] = 1

    criterion = nn.CrossEntropyLoss()

    net = nn.Net([
        nn.Linear(784, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 10),
    ])

    optimizer = nn.SGD(net.parameters(), 0.1, 0.9, weight_decay=0.0001)

    bs = 100

    t0 = time.time()
    for i in range(10):
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)

        losses = []
        for s in range(len(x_train) // bs):
            x = x_train[indices[s*bs:(s+1)*bs]]
            y = y_onehot[indices[s*bs:(s+1)*bs]]

            z = net.forward(x)
            loss = criterion.forward(z, y)
            dloss = criterion.backward()
            d = net.backward(dloss)

            optimizer.step()

            losses.append(loss)

        acc_train = get_accuracy(net, x_train, y_train) * 100
        acc_test = get_accuracy(net, x_test, y_test) * 100

        print('Epoch %4d -> loss %7.3f train %7.3f / test %7.3f' %
              (i, np.mean(losses), acc_train, acc_test))
    t1 = time.time()
    elapsed = t1 - t0
    print('Elapsed -> %.3f s' % elapsed)


def test_grad_cupy():
    np.random.seed(1234)

    train_image, train_label, test_image, test_label = mnist.load()

    x_train = train_image / 255.
    y_train = train_label
    x_test = test_image / 255.
    y_test = test_label

    y_onehot = np.zeros((len(y_train), 10), float)
    y_onehot[np.arange(len(y_train)), y_train] = 1

    x_train = cp.asarray(x_train)
    y_train = cp.asarray(y_train)
    x_test = cp.asarray(x_test)
    y_test = cp.asarray(y_test)
    y_onehot = cp.asarray(y_onehot)

    criterion = nncu.CrossEntropyLoss()

    net = nncu.Net([
        nncu.Linear(784, 512),
        nncu.LeakyReLU(),
        nncu.Linear(512, 10),
    ])

    optimizer = nncu.SGD(net.parameters(), 0.1, 0.9, weight_decay=0.0001)

    bs = 100
    t0 = time.time()
    for i in range(10):
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)

        losses = []
        for s in range(len(x_train) // bs):
            x = x_train[indices[s*bs:(s+1)*bs]]
            y = y_onehot[indices[s*bs:(s+1)*bs]]

            z = net.forward(x)
            loss = criterion.forward(z, y)
            dloss = criterion.backward()
            d = net.backward(dloss)

            optimizer.step()

            losses.append(loss)

        acc_train = get_accuracy_cupy(net, x_train, y_train) * 100
        acc_test = get_accuracy_cupy(net, x_test, y_test) * 100

        print('Epoch %4d -> loss %7.3f train %7.3f / test %7.3f' %
              (i, cp.asarray(losses).mean(), acc_train, acc_test))
    t1 = time.time()
    elapsed = t1 - t0
    print('Elapsed -> %.3f s' % elapsed)


if __name__ == '__main__':
    test_grad()
    test_grad_cupy()
