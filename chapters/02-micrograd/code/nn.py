"""
A tiny neural-network library built on top of our `Value` engine.
=================================================================
Because every `Value` knows how to backprop itself, we can build a whole neural
network out of Values and get training "for free": call `loss.backward()` and every
weight's `.grad` is filled in automatically. This mirrors how PyTorch's `nn.Module`
sits on top of its autograd engine — just thousands of times smaller.

    Neuron  →  one weighted sum + a nonlinearity
    Layer   →  a row of independent Neurons
    MLP     →  a stack of Layers (a "multi-layer perceptron")
"""

import random
from engine import Value


class Module:
    """Shared base class: lets us zero gradients and list parameters."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    """Computes tanh(w · x + b): a weighted sum of inputs, then a squash."""

    def __init__(self, nin):
        # one weight per input, plus a bias — all random to start
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w · x + b  (start the sum at the bias), then tanh
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    """A row of `nout` neurons, each seeing the same inputs."""

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs   # unwrap a single output

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """A stack of layers. MLP(3, [4, 4, 1]) = 3 inputs → 4 → 4 → 1 output."""

    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)        # the output of one layer is the input to the next
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
