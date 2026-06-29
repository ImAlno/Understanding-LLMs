"""
micrograd engine — a tiny scalar-valued automatic-differentiation engine.
=========================================================================
In Chapter 1 we called `loss.backward()` and treated it as magic. This file *is*
that magic, in ~90 lines.

The idea: instead of storing a plain number, we wrap each number in a `Value`
object that also remembers **how it was computed** (which numbers and which
operation produced it). That record is a little graph. Once we have the graph, we
can walk it backwards and compute, for every number in it, the **gradient** — how
much the final output would change if we nudged that number a hair. That backward
walk is **backpropagation**, and it's the entire reason neural nets can learn.

This is a faithful, beginner-commented version of Andrej Karpathy's `micrograd`.
"""

import math


class Value:
    """Stores a single number and the graph that produced it."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data                 # the actual number
        self.grad = 0.0                  # d(final output)/d(self); 0 until backward() runs
        # --- the bookkeeping that makes autograd work ---
        self._backward = lambda: None    # a function that pushes our grad to our inputs
        self._prev = set(_children)      # the Value(s) this one was built from
        self._op = _op                   # the op that made us (e.g. '+', '*') — for printing

    # ---------- forward operations (each also wires up its own backward) ----------

    def __add__(self, other):            # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # addition passes the gradient straight through to both inputs
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):            # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # product rule: d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):            # self ** k  (k is a constant)
        assert isinstance(other, (int, float)), "only constant int/float powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # power rule: d(x^k)/dx = k * x^(k-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):                       # e ** self
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # d(e^x)/dx = e^x, which is exactly out.data
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):                      # a smooth "squash to (-1, 1)" nonlinearity
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):                      # the other classic nonlinearity: max(0, x)
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    # ---------- the backward pass ----------

    def backward(self):
        """Fill in `.grad` for every Value that fed into this one."""
        # 1) put the graph in topological order (every node after its inputs)
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 2) the output's gradient w.r.t. itself is 1, then apply the chain rule
        #    backwards through the graph, one node at a time.
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # ---------- niceties so Python operators "just work" ----------

    def __neg__(self):        return self * -1
    def __radd__(self, other): return self + other          # other + self
    def __sub__(self, other):  return self + (-other)        # self - other
    def __rsub__(self, other): return other + (-self)        # other - self
    def __rmul__(self, other): return self * other           # other * self
    def __truediv__(self, other):  return self * other ** -1  # self / other
    def __rtruediv__(self, other): return other * self ** -1  # other / self

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
