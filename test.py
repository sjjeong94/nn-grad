import time
import numpy as np
import torch

import nn
import mnist


def get_accuracy(model, x, y):
    x = model.forward(x)
    y_pred = np.argmax(x, axis=1)
    acc = (y_pred == y).mean()
    return acc


def get_accuracy_torch(model, x, y):
    with torch.no_grad():
        x = model(x)
        y_pred = torch.argmax(x, axis=1)
        acc = (y_pred == y).sum() / len(x)
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

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    y_onehot = y_onehot.astype(np.float32)

    criterion = nn.CrossEntropyLoss()

    net = nn.Net([
        nn.Linear(784, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 10),
    ])

    optimizer = nn.SGD(net.parameters(), 0.1, 0.9, weight_decay=0.0001)

    bs = 100

    t0 = time.time()
    for i in range(5):
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


def test_torch():
    np.random.seed(1234)

    train_image, train_label, test_image, test_label = mnist.load()

    x_train = train_image / 255.
    y_train = train_label
    x_test = test_image / 255.
    y_test = test_label

    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)

    criterion = torch.nn.CrossEntropyLoss()

    net = torch.nn.Sequential(
        torch.nn.Linear(784, 512),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(512, 10),
    )

    optimizer = torch.optim.SGD(
        net.parameters(), 0.1, 0.9, weight_decay=0.0001)

    bs = 100

    t0 = time.time()
    for i in range(5):
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)

        losses = 0
        for s in range(len(x_train) // bs):
            x = x_train[indices[s*bs:(s+1)*bs]]
            y = y_train[indices[s*bs:(s+1)*bs]]

            net.zero_grad()
            z = net(x)
            loss = criterion(z, y)
            loss.backward()

            optimizer.step()

            losses += loss.detach()

        acc_train = get_accuracy_torch(net, x_train, y_train) * 100
        acc_test = get_accuracy_torch(net, x_test, y_test) * 100

        losses = losses / (len(x_train) // bs)
        print('Epoch %4d -> loss %7.3f train %7.3f / test %7.3f' %
              (i, losses, acc_train, acc_test))
    t1 = time.time()
    elapsed = t1 - t0
    print('Elapsed -> %.3f s' % elapsed)


if __name__ == '__main__':
    test_grad()
    test_torch()
