import numpy as np


def softmax(z):
    z = np.exp(z)
    return z / np.sum(z, axis=1, keepdims=True)


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

    def update(self, lr=1):
        return


class Linear:
    def __init__(self, in_ch, out_ch):
        r = np.sqrt(1 / in_ch)
        self.w = np.random.uniform(-r, r, (in_ch, out_ch))
        self.b = np.random.uniform(-r, r, out_ch)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dz):
        self.w_grad = np.dot(dz.T, self.x).T
        self.b_grad = np.sum(dz, axis=0)
        return np.dot(dz, self.w.T)

    def update(self, lr=1):
        self.w = self.w - lr * self.w_grad
        self.b = self.b - lr * self.b_grad


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

    def update(self, lr=1):
        for layer in self.layers:
            layer.update(lr=lr)
