import numpy as np


def softmax(z):
    z = np.exp(z)
    return z / np.sum(z, axis=1, keepdims=True)


class Tensor:
    def __init__(self, array):
        self.data = array


class SGD:
    def __init__(self, parameters, lr=0.1, momentum=0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        b = []
        for parameter in self.parameters:
            b.append(np.zeros(parameter.data.shape))
        self.b = b

    def step(self):
        lr = self.lr
        for i, parameter in enumerate(self.parameters):
            #parameter.data = parameter.data - lr * parameter.grad
            self.b[i] = self.momentum * self.b[i] + parameter.grad
            parameter.data = parameter.data - lr * self.b[i]


class CrossEntropyLoss:
    def __init__(self):
        return

    def forward(self, z, y):
        self.s = softmax(z)
        self.y = y

        loss = -np.sum(y * np.log(self.s + 1e-9)) / len(self.y)
        return loss

    def backward(self):
        return (self.s - self.y) / len(self.y)


class ReLU:
    def __init__(self):
        return

    def forward(self, z):
        self.z = z
        return np.maximum(0, z)

    def backward(self, da):
        return da * (self.z > 0)

    def parameters(self):
        return []


class Linear:
    def __init__(self, in_ch, out_ch):
        r = np.sqrt(1 / in_ch)
        self.w = Tensor(np.random.uniform(-r, r, (in_ch, out_ch)))
        self.b = Tensor(np.random.uniform(-r, r, out_ch))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w.data) + self.b.data

    def backward(self, dz):
        self.w.grad = np.dot(dz.T, self.x).T
        self.b.grad = np.sum(dz, axis=0)
        return np.dot(dz, self.w.data.T)

    def parameters(self):
        return [self.w, self.b]


class Net:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[-i-1].backward(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
