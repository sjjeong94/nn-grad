import numpy as np


def softmax(z):
    z = np.exp(z)
    return z / np.sum(z, axis=1, keepdims=True)


class Tensor:
    def __init__(self, array):
        self.data = array


class SGD:
    def __init__(self, parameters, lr=0.1, momentum=0, weight_decay=0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.b, self.g = [], []
        for parameter in self.parameters:
            self.b.append(np.zeros(parameter.data.shape).astype(np.float32))
            self.g.append(np.zeros(parameter.data.shape).astype(np.float32))

    def step(self):
        lr = self.lr
        for i, parameter in enumerate(self.parameters):
            g = parameter.grad
            if self.weight_decay > 0:
                g = g + self.weight_decay * parameter.data
            if self.momentum > 0:
                self.b[i] = self.momentum * self.b[i] + g
                g = self.b[i]
            parameter.data = parameter.data - lr * g
            self.g[i] = g


class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return


class CrossEntropyLoss(Module):
    def __init__(self):
        self.eps = np.float32(1e-9)

    def forward(self, z, y):
        self.s = softmax(z)
        self.y = y

        loss = -np.sum(y * np.log(self.s + self.eps))
        loss = loss / np.float32(len(self.y))
        return loss

    def backward(self):
        return (self.s - self.y) / np.float32(len(self.y))


class Sigmoid(Module):
    def __init__(self):
        return

    def forward(self, z):
        a = 1 / (1+np.exp(-z))
        self.a = a
        return a

    def backward(self, da):
        return da * self.a * (1-self.a)

    def parameters(self):
        return []


class ReLU(Module):
    def __init__(self):
        return

    def forward(self, z):
        self.z = z
        return np.maximum(0, z)

    def backward(self, da):
        return da * (self.z > 0)

    def parameters(self):
        return []


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        self.ns = np.float32(negative_slope)
        return

    def forward(self, z):
        self.z = z
        return np.maximum(0, z) + (z < 0) * self.ns * z

    def backward(self, da):
        return da * ((self.z > 0) + (self.z < 0) * self.ns)

    def parameters(self):
        return []


class Linear(Module):
    def __init__(self, in_ch, out_ch):
        r = np.sqrt(1 / in_ch)
        self.w = Tensor(
            np.random.uniform(-r, r, (in_ch, out_ch)).astype(np.float32))
        self.b = Tensor(np.random.uniform(-r, r, out_ch).astype(np.float32))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w.data) + self.b.data

    def backward(self, dz):
        self.w.grad = np.dot(self.x.T, dz)
        self.b.grad = np.sum(dz, axis=0)
        return np.dot(dz, self.w.data.T)

    def parameters(self):
        return [self.w, self.b]


class Net(Module):
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
