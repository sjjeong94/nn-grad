import cupy as cp


def softmax(z):
    z = cp.exp(z)
    return z / cp.sum(z, axis=1, keepdims=True)


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
            self.b.append(cp.zeros(parameter.data.shape))
            self.g.append(cp.zeros(parameter.data.shape))

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


class CrossEntropyLoss:
    def __init__(self):
        return

    def forward(self, z, y):
        self.s = softmax(z)
        self.y = y

        loss = -cp.sum(y * cp.log(self.s + 1e-9)) / len(self.y)
        return loss

    def backward(self):
        return (self.s - self.y) / len(self.y)


class Sigmoid:
    def __init__(self):
        return

    def forward(self, z):
        a = 1 / (1+cp.exp(-z))
        self.a = a
        return a

    def backward(self, da):
        return da * self.a * (1-self.a)

    def parameters(self):
        return []


class ReLU:
    def __init__(self):
        return

    def forward(self, z):
        self.z = z
        return cp.maximum(0, z)

    def backward(self, da):
        return da * (self.z > 0)

    def parameters(self):
        return []


class LeakyReLU:
    def __init__(self):
        return

    def forward(self, z):
        self.z = z
        return cp.maximum(0, z) + (z < 0) * 0.01 * z

    def backward(self, da):
        return da * ((self.z > 0) + (self.z < 0) * 0.01)

    def parameters(self):
        return []


class Linear:
    def __init__(self, in_ch, out_ch):
        r = cp.sqrt(1 / in_ch)
        self.w = Tensor(cp.random.uniform(-r, r, (in_ch, out_ch)))
        self.b = Tensor(cp.random.uniform(-r, r, out_ch))

    def forward(self, x):
        self.x = x
        return cp.dot(x, self.w.data) + self.b.data

    def backward(self, dz):
        self.w.grad = cp.dot(self.x.T, dz)
        self.b.grad = cp.sum(dz, axis=0)
        return cp.dot(dz, self.w.data.T)

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
