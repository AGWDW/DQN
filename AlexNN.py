import numpy as np
from collections import namedtuple
import random as rnd
from scipy.special import softmax
np.set_printoptions(precision=5)

NNFunc = namedtuple(
    'NNFunc',
    ('function', 'derivative')
)

MemoryElement = namedtuple(
    'MemoryElement',
    ('state', 'action', 'reward', 'resulting_state')
)


def to_data(sample):
    return MemoryElement(*zip(*sample))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu_d(x):
    t = np.copy(x)
    x[x <= 0] = 0
    x[x > 0] = 1
    r = np.isnan(x).astype(float)
    if np.max(r) == 1:
        raise Exception('fd')
    return x


def to_activation(name):
    res = NNFunc(None, None)
    if name == 'sigmoid':
        res = NNFunc(sigmoid, lambda x: sigmoid(x) * sigmoid(-x))
    elif name == 'softmax':
        res == NNFunc(softmax, lambda x: x)
        pass
    elif name == 'relu':
        res = NNFunc(lambda x: np.maximum(np.zeros_like(x), x), relu_d)
        pass
    else:  # pass through
        res = NNFunc(lambda x: x, lambda x: 1)
    return res


def to_cost_func(name):
    res = NNFunc(None, None)
    if name == 'mse':
        res = NNFunc(lambda x, y: 0.5 * np.linalg.norm(x - y) ** 2, lambda x, y: x - y)
    elif name == 'ce':
        pass
    else:
        return to_cost_func('mse')
    return res


class DQN:
    def __init__(self, shape, activations, cost_func):
        if len(activations) < len(shape):
            while len(activations) < len(shape):
                activations.append('')
        elif len(activations) > len(shape):
            raise Exception('size of activations doesnt match the size of the neural net')
        self.shape = shape
        self.W = [np.random.randn(y, x) for x, y in zip(shape[:-1], shape[1:])]
        self.B = [np.random.randn(y, 1) for y in shape[1:]]
        self.cost = to_cost_func(cost_func)

        self.xs = []
        self.zs = []

        self.activations = [to_activation(act) for act in activations]
        self.num_layers = len(shape)

    def forward(self, x, show=False):
        y = x
        self.xs = []
        self.zs = [x]
        for w, b, a in zip(self.W, self.B, self.activations):
            if show:
                print(f'{w.shape} . {y.shape} + {b.shape}')
            x = np.dot(w, y) + b
            y = a.function(x)
            self.xs.append(x)
            self.zs.append(y)

        y = self.activations[-1].function(x)
        return y

    def SGD(self, epochs, batch_size, training_data, lr):
        for epoch in range(epochs):
            rnd.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)

    def update_mini_batch(self, mini_batch, lr):
        nabla_b = [np.zeros_like(b) for b in self.B]
        nabla_w = [np.zeros_like(w) for w in self.W]
        for x, y in mini_batch:
            # print(f"x: {x} y: {y}")
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # updates the weights and biases
        self.update_w_b(mini_batch, nabla_w, nabla_b, lr)

    def update_w_b(self, batch_size, nabla_w, nabla_b, lr):
        # updates the weights and biases
        self.W = [w - (lr / len(batch_size)) * nw for w, nw in zip(self.W, nabla_w)]
        self.B = [b - (lr / len(batch_size)) * nb for b, nb in zip(self.B, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros_like(b) for b in self.B]
        nabla_w = [np.zeros_like(w) for w in self.W]

        # fwd pass
        self.forward(x)

        # backward pass
        delta = self.cost.derivative(self.zs[-1], y) * self.activations[-1].derivative(self.xs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, self.zs[-2].transpose())
        for l in range(2, self.num_layers):
            z = self.xs[-l]
            # print(f"z: {z}")
            sp = self.activations[-l].derivative(z)
            delta = np.dot(self.W[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, self.zs[-l - 1].transpose())
        return nabla_b, nabla_w

    def copy(self):
        res = DQN((1, 1), ['', ''], 'mse')
        res.W = self.W
        res.B = self.B
        res.shape = self.shape
        res.activations = self.activations
        res.cost = self.cost
        res.num_layers = self.num_layers
        return res
