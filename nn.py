import numpy as np


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
        self.w = np.random.randn(in_ch, out_ch)
        self.b = np.random.randn(out_ch)

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
