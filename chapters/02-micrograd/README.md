# Chapter 2 — Micrograd: Backpropagation from Scratch

> *Opening the black box.* In Chapter 1 we typed `loss.backward()`, called it
> "magic," and promised to explain it later. This is later. In this chapter we build
> that one line **ourselves**, from scratch, in about 90 lines of plain Python — a
> tiny **automatic-differentiation engine**. Once you've built it, you'll understand
> the single mechanism that lets *every* neural network — from our bigram model to
> ChatGPT — actually learn.

**You will be able to:**
- explain what a **derivative** and a **gradient** really are, with zero calculus;
- build a `Value` object that records how each number was computed (a *graph*);
- understand the **chain rule** and use it to do **backpropagation by hand**;
- automate it so any expression can compute its own gradients;
- build and train a small neural network with your own engine — and see it learn;
- recognize that this is *exactly* what PyTorch's autograd does under the hood.

**Prerequisites:** [Chapter 1](../01-bigram/) (you should recognize the
forward → backward → update loop), and basic Python **including classes** — we explain
the class features we use as we hit them. **No calculus required:** we build every
bit of math intuition we need from scratch.

**Time:** ~2–3 hours if you build along. **Hardware:** any laptop. No GPU, no dataset.

> ### 📓 Build it yourself in the notebook
> As in Chapter 1, the companion notebook **[`code/explore.ipynb`](./code/explore.ipynb)**
> is where the learning happens. You'll do the two things that make backprop *click* —
> compute a derivative by hand, and **backpropagate through a graph yourself** — then
> assemble the engine. Each "✍️ Your turn" cell gives a hint with the answer one click
> away, and a **"Check your work"** cell tells you instantly if you nailed it. Read a
> section here, then build it there. Watch for: ✏️ **In the notebook → Step N**.

---

## 0. Setup

No dataset this time — just Python. If you set up the environment in Chapter 1 you're
already done. Otherwise, from the repo root:

```bash
uv venv --python 3.13 .venv && uv pip install -r requirements.txt
.venv/bin/jupyter lab        # then open chapters/02-micrograd/code/explore.ipynb
```

The finished reference code lives in [`code/engine.py`](./code/engine.py) (the engine),
[`code/nn.py`](./code/nn.py) (the neural net), and
[`code/train_demo.py`](./code/train_demo.py) (a training run) — for checking your work.

---

## A 2-minute primer: derivatives

The one new idea in this chapter is the **derivative**, and it's simpler than school
made it sound. A derivative answers one question:

> *If I nudge this input up by a tiny amount, how much does the output change?*

That's it — a derivative is a **sensitivity**. Let's see it with actual numbers.
Take `f(x) = x²` and sit at `x = 3`, so `f = 9`. Now nudge `x` up by a tiny
`h = 0.001`:

```
f(3.001) = 3.001 × 3.001 = 9.006001
change in f = 9.006001 − 9 = 0.006001
change in f per unit of nudge = 0.006001 / 0.001 ≈ 6
```

So at `x = 3`, the output changes about **6 times** as fast as we wiggle the input.
We say "the derivative of `f` at `x = 3` is 6." (If you remember the rule
`d(x²)/dx = 2x`, note `2 × 3 = 6` — same answer. But we never needed the rule; we just
nudged and measured. That trick is called a **finite difference**, and you'll use it
in the notebook.)

**The sign matters as much as the size.** Our derivative at `x = 3` was *positive* (`+6`): nudge
`x` up, and `f` goes up. Now sit at `x = −3` instead. Nudging up to `−2.999` gives
`f = 8.994001`, a change of `−0.005999` per `0.001` of nudge — the derivative is *negative*
(`−6`): nudge `x` up, and `f` goes *down*. And right at the bottom, `x = 0`, a tiny nudge barely
moves `f` at all — the derivative is `0`. Picture the curve of `x²`, a valley:

```
  f(x) = x²    ╲                         ╱
                ╲  slope −6 here ↙     ↘ slope +6 here
                 ╲____             ____╱
                      ╲___  ___  __╱
                          ╲╱   ╲╱   ← slope 0 at the bottom (x = 0)
```

A derivative is just the **slope** of the curve at a point: rising steeply → big positive, falling
steeply → big negative, flat (a minimum) → zero. *This is the whole reason gradient descent
works.* To make the loss smaller you step **opposite** the derivative — downhill — by an amount
proportional to its size; where the derivative reaches zero, the slope is flat and you've arrived
at the bottom. The gradient is your compass: its **sign** says which way is downhill, its **size**
says how steep.

> 📐 **Notation, decoded once.** You'll see derivatives written `d(out)/da` throughout
> this chapter. Read the whole thing as one symbol meaning *"how much `out` moves when
> you nudge `a`"* — exactly the sensitivity above, just in shorthand. (The `d`s aren't
> separate numbers; `d(out)/da` is a single name for "the derivative of `out` with
> respect to `a`.")

**Why we care.** In Chapter 1, "the gradient" was the list of numbers telling us which
way to nudge each weight to lower the loss. Now you can see exactly what each of those
numbers *is*: the derivative of the loss with respect to that weight — *how sensitive
the loss is to that one knob*. Backpropagation, this whole chapter, is just an
efficient way to compute that sensitivity for **every** knob at once.

---

## 1. The plan: a number that remembers

Here's the obstacle. A plain Python number has no memory:

```python
a = 2.0
b = -3.0
c = a * b      # c is now -6.0 ... and it has completely forgotten it came from a and b
```

To compute gradients, we need the opposite: every result must remember **what it was
built from** and **which operation built it**. So we'll wrap each number in a small
object — a `Value` — that carries that history along. A chain of `Value`s forms a
**computational graph**: a record of the whole calculation that we can later walk
*backwards* to find every derivative.

> ✏️ **In the notebook → Steps 1–2.** First you'll measure a derivative by nudging,
> then backpropagate a tiny graph by hand — the two "aha" moments.

---

## 2. The `Value` object

Here is the skeleton — just storage, no math yet:

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data                 # the actual number
        self.grad = 0.0                  # d(final output)/d(self) — 0 until we backprop
        self._backward = lambda: None    # how to pass our gradient back to our inputs
        self._prev = set(_children)      # the Values we were built from
        self._op = _op                   # the operation that made us (e.g. '*')
```

Four ideas live in there:
- **`data`** — the number itself.
- **`grad`** — the gradient, *how much the final output changes if we nudge this
  Value*. It starts at `0` and gets filled in during the backward pass.
- **`_prev`** — the Value's parents in the graph (what it was computed from). The
  leading underscore is a Python convention meaning "internal — don't poke at this."
- **`_backward`** — a little function, set when the Value is created, that knows how to
  hand this Value's gradient back to its parents. (A function stored *inside* an object
  may feel odd; it's the trick that makes the whole engine tick, and it'll make sense
  in §6.)

*New Python in that snippet, if it's unfamiliar:* `lambda: None` is a tiny throwaway
function that takes no arguments and returns `None` — just a placeholder we'll replace
later. `_children=()` defaults to an empty **tuple**. And `set(_children)` stores the
parents as a **set** (no duplicates, fast "is this in here?" checks).

---

## 3. The forward pass: building the graph

Now we teach `Value` to do arithmetic. The rule for every operation is the same:
compute the result, **and** record the parents and the op. Here's `+` and `*`:

```python
    def __add__(self, other):                       # called when you write a + b
        out = Value(self.data + other.data, (self, other), "+")
        return out

    def __mul__(self, other):                       # called when you write a * b
        out = Value(self.data * other.data, (self, other), "*")
        return out
```

(`__add__` and `__mul__` are Python's hooks for the `+` and `*` symbols — defining them
lets us write `a + b` on our own objects.) Now a calculation leaves a trail:

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a * b          # e remembers parents (a, b) and op '*'
d = e + c          # d remembers parents (e, c) and op '+'
```

`d` is the root of a little graph: `d = (a × b) + c`. Drawn out:

```
a(2.0) ─┐
        ├─[*]→ e(-6.0) ─┐
b(-3.0)─┘               ├─[+]→ d(4.0)
              c(10.0) ──┘
```

We built the graph going **forward** (left to right). To get gradients we'll walk it
**backward** (right to left). That backward walk is the whole game.

---

## 4. The chain rule: the heart of backprop

We want `d.grad`, `e.grad`, `a.grad`, … — the sensitivity of the final output to every
node. The tool that gets us there is the **chain rule**, and the intuition is just:
**sensitivities multiply along a chain.**

> Imagine a chain of gears. If turning gear A one notch turns gear B **two** notches,
> and one notch of B moves gear C **five** notches, then one notch of A moves C
> **2 × 5 = 10** notches. Sensitivities multiply.

Concretely, with numbers: say `b = a²` and `c = 3b`, sitting at `a = 4`. The two *local*
sensitivities are `d(b)/da = 2a = 8` (nudge `a`, `b` moves 8×) and `d(c)/db = 3` (nudge `b`, `c`
moves 3×). So nudging `a` should move `c` by `8 × 3 = 24`. Check it directly: at `a = 4`,
`c = 3·16 = 48`; at `a = 4.001`, `c = 3·16.008001 = 48.024003`, a change of `0.024` per `0.001` of
nudge — that's **24**. The chain rule (multiply the locals) and the brute-force nudge agree exactly.

In our graph, `a` affects `e` affects `d`. So:

```
d(output)/d(a)  =  d(output)/d(e)  ×  d(e)/d(a)
   (how a affects the output)  =  (how e affects output)  ×  (how a affects e)
```

The second factor, `d(e)/d(a)`, is a **local** derivative — it only involves one
operation (`e = a * b`), so it's easy. The first factor is the gradient that has
*already* flowed back to `e` from the output. **Backprop = at each node, multiply the
gradient that arrived from above by the node's own local derivative, and pass the
result to its parents.**

Each operation has a dead-simple local rule:

| Operation | Local derivative | In words |
|-----------|------------------|----------|
| `out = a + b` | `d(out)/da = 1`, `d(out)/db = 1` | addition just **passes the gradient through** |
| `out = a * b` | `d(out)/da = b`, `d(out)/db = a` | each input's grad is the **other** input's value |
| `out = a ** k` | `d(out)/da = k·a^(k−1)` | the power rule |
| `out = tanh(a)` | `d(out)/da = 1 − tanh(a)²` | (we'll meet tanh in §7) |

That's the whole toolkit. Everything else is bookkeeping.

---

## 5. Backprop by hand (the "aha")

> ✏️ **In the notebook → Step 2 — do this one by hand.** It is the single most
> important exercise in the chapter.

Let's backprop our graph `d = (a × b) + c` with `a=2, b=−3, c=10`, so `e=−6, d=4`. We
go **right to left**:

1. **Start at the output.** `d.grad = 1` — nudging `d` changes `d` exactly one-for-one
   with itself.
2. **Through the `+` node** (`d = e + c`). Addition passes gradient straight through:
   `e.grad = 1`, `c.grad = 1`.
3. **Through the `*` node** (`e = a × b`). Each input gets the *other* input's value,
   times the gradient arriving at `e`:
   - `a.grad = b.data × e.grad = −3 × 1 = −3`
   - `b.grad = a.data × e.grad =  2 × 1 =  2`

So `a.grad = −3`: if we increase `a` a touch, `d` goes *down* at 3× the rate. **Sanity
check by nudging:** with `a = 2.001`, `e = 2.001 × −3 = −6.003`, `d = 4 − 0.003`, a
change of `−0.003` per `0.001` of nudge = `−3`. ✅ The chain rule and the finite
difference agree. That agreement *is* the moment backprop becomes real.

---

## 6. Automating it: `_backward`, topological order, `backward()`

Doing that by hand is great for understanding, hopeless at scale (GPT has billions of
nodes). So we automate it. The trick: when each operation builds its `out`, it also
stores a tiny `_backward` function that knows that node's local rule. Here are `+` and
`*` again, now complete:

```python
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad  += 1.0 * out.grad     # pass gradient straight through
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad  += other.data * out.grad   # each input gets the OTHER's value
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out
```

Two things every beginner asks about here:

- **Why `+=` and not `=`?** Because a Value can feed into *several* places, and its total
  sensitivity is the **sum** of the gradients arriving along each path. The cleanest case is a value
  used twice — `g = a * a` (that's `a²`). Here `a` is *both* inputs to the `*`, so backprop reaches
  it from two sides and adds: at `a = 3` with `g.grad = 1`, `a.grad = a·g.grad + a·g.grad =
  3 + 3 = 6` — exactly `d(a²)/da = 2a = 6`. With a plain `=`, the second contribution would
  *overwrite* the first and you'd get `3`, the wrong answer. Accumulation is what makes the two
  paths add up. (It's also why we zero the grads before each backward pass — Chapter 1's
  `W.grad = None` was the same idea; otherwise last step's gradients would leak into this one.)
- **Why is `_backward` a function stored on `out`?** So we can call it later, once
  `out.grad` is known. Each node says "when you finally know *my* gradient, here's how
  I'll pass it to my parents." This works because of a Python feature called a
  **closure**: a function defined *inside* another function keeps a live link to the
  variables around it, even after the outer function has returned. Tiny example:

  ```python
  def make_adder(n):
      def add(x):
          return x + n        # `add` still remembers `n`...
      return add
  add5 = make_adder(5)
  add5(10)                    # ...so this is 15 — `add` captured n = 5
  ```

  Our `_backward` is exactly this pattern: defined inside `__mul__`, it captures `self`,
  `other`, and `out`. So later — long after `__mul__` has finished running — calling
  `out._backward()` still knows *which three Values* to update. That capture is the
  whole trick.

Now the driver. To backprop the whole graph we must process nodes in the right order:
a node can only push its gradient back *after* its own gradient is complete — i.e.
after everything downstream of it is done. The standard way to get that order is a
**topological sort** (every node comes after its parents); we then walk it in reverse:

```python
    def backward(self):
        topo, visited = [], set()
        def build_topo(v):                 # depth-first: add a node after its parents
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0                    # seed: d(output)/d(output) = 1
        for v in reversed(topo):           # walk output → inputs
            v._backward()                  # each node hands its gradient to its parents
```

`build_topo` is **recursive** — it calls *itself* on each child before adding the
current node to the list. (If you've never seen a function call itself, that's
*recursion*; here, trust that it visits every node in the graph exactly once.) Because a
node is appended only *after* all its parents are, the list comes out with parents always
before children — for our 4-node graph `d = (a×b) + c`, an order like
`[c, a, b, e, d]`. We then walk it **`reversed`** — `d` first, then back toward the
inputs — so each node's `_backward` runs only once its own `.grad` is final. That
ordering is the only thing standing between us and a tangle of gradients computed in the
wrong order.

**Watch one run, node by node.** On our graph `d = (a×b) + c`, `build_topo` produces the order
`[c, a, b, e, d]`, and we walk it **reversed** (`d` first):

```
seed:  d.grad = 1
d._backward()   (the + node):   e.grad += 1·1 = 1        c.grad += 1·1 = 1
e._backward()   (the * node):   a.grad += b·e.grad = −3·1 = −3
                                b.grad += a·e.grad =  2·1 =  2
a, b, c are leaves — their _backward does nothing (no parents to hand a gradient to)
```

Every `.grad` is now filled: `d=1, e=1, c=1, a=−3, b=2` — *exactly* the numbers you found by hand
in §5. Watch the ordering earn its keep: `e._backward` ran only *after* `d._backward` had given
`e` its gradient, so by the time `e` passed gradient to `a` and `b`, its own `.grad` was already
final. Swap in a million nodes and nothing changes but the length of the list.

Call `d.backward()` and every `.grad` in the graph fills in — the same numbers you got
by hand, for a graph of any size. **That's the entire engine** — and with the closure and
the ordering both spelled out, there's no magic left in it. Everything after this is just
*using* it.

> 💡 **Why backprop, and not just nudge every weight?** You *could* get each weight's gradient the
> finite-difference way from the primer — nudge that one weight, re-run the whole forward pass,
> measure the change in the loss. But a real network has *millions* of weights, and that recipe
> needs a separate forward pass **per weight**: millions of forward passes for a single gradient
> step. Hopeless. Backprop's whole magic is that **one** backward walk fills in the gradient for
> **every** weight at once — reusing each node's gradient as it flows down — for about the cost of
> a *single* forward pass, not millions. That efficiency is the entire reason training large
> networks is even possible. (Christopher Olah's article, linked at the end, draws this out
> beautifully.)

---

## 7. A nonlinearity: `tanh`

To build a neuron we need one more ingredient: a **nonlinearity**. Why? Because a stack of purely
linear steps (adds and multiplies) collapses into a single linear step. See it concretely: send
`x` through `y = 2x + 1`, then send that through `z = 3y − 4`. Substitute the first into the
second: `z = 3(2x + 1) − 4 = 6x − 1` — a *single* straight line. Stack a hundred more linear layers
and you'd **still** just get `z = (some number)·x + (some other number)`. No amount of stacking
buys you a curve, so all those extra layers are wasted.

A nonlinear **squash** between the linear steps breaks that collapse: `tanh(2x + 1)` genuinely
*cannot* be flattened back into `m·x + b`, so now each layer adds real expressive power — that's
what lets a network bend to fit complicated shapes.

We'll use **`tanh`**, which gently squashes any number into the range `(−1, 1)`: big positives →
near `+1`, big negatives → near `−1`, and `0 → 0`. Picture its S-shape:

```
 tanh(x) +1┤           ╭─────────────
           ┤       ╭──╯
          0┤───────┼──╯
           ┤    ╭─╯
        −1 ┤────╯
           └──────────┼──────────────
                      0        x
```

Near `0` it's almost a straight line — it passes small signals through nearly untouched. Far out
in either direction it **saturates**: it flattens toward ±1, where its slope `1 − tanh²` shrinks
toward `0`. (Hold onto that — a *saturated* neuron has almost no gradient, so it barely learns;
that's a real failure mode we diagnose and fix in Chapter 7.) Its local derivative is beautifully
simple, `1 − tanh(x)²`:

```python
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
```

Notice the **shape is identical for every operation** — `+`, `*`, `tanh`, and any other:

1. compute the forward result and wrap it in a `Value`, recording its parents and the op;
2. define a `_backward` closure holding *this op's local rule* (how each parent's grad relates to
   `out.grad`);
3. attach it as `out._backward` and return.

That's the whole recipe. Want `exp`, division, or a different squash like `ReLU`? Same three
steps — just drop in that operation's local derivative. The engine doesn't get *more complex* as
you add operations, it just learns more *verbs*. (You'll add a couple yourself in the exercises and
gradient-check them against PyTorch — proof the pattern really does generalize.)

> ✏️ **In the notebook → Steps 3–4.** You'll fill in the `_backward` rules for `+`
> and `*`, then build a neuron's forward pass.

---

## 8. From `Value` to a neural network

A **neuron** is almost nothing: multiply each input by a weight, add a bias, squash.
Because every piece is a `Value`, the neuron is automatically differentiable.

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # a weight per input
        self.b = Value(random.uniform(-1, 1))                        # one bias
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)    # w·x + b
        return act.tanh()                                            # squash
```

Two bits of Python worth unpacking there:
- **`__call__`** is the dunder that makes an object *callable* — defining it lets us
  write `n(x)`, which is how we'll use neurons (and whole models: `model(x)`) later.
- **`w · x`** is the **dot product**: multiply each weight by its matching input and add
  the results. That dense one-liner is just the dot product written compactly — `zip`
  pairs each weight with its input, and starting the `sum` at `self.b` folds the bias in.
  Spelled out as an ordinary loop, it's:

  ```python
  act = self.b
  for wi, xi in zip(self.w, x):
      act = act + wi * xi          # add weight × input, one at a time
  return act.tanh()
  ```

**What is a neuron, really?** `w·x + b` is a *weighted vote*: it scores how strongly the input `x`
lines up with the neuron's own pattern of weights `w`, shifted by the bias `b`. A big positive
score means "`x` looks like the thing I'm tuned to detect"; a big negative score means "the
opposite"; near zero means "on the fence." The `tanh` then turns that raw score into a soft yes/no
in `(−1, 1)`. So **one neuron draws one soft boundary** through its input space — and stacking many
of them, in layers, lets the network carve that space into the intricate regions real data needs.
The weights and bias are the knobs; backprop is what tunes them.

**Let's backprop one by hand**, to watch all three rules fire together. Take the simplest neuron —
a single input — `n = tanh(w·x + b)`, with `w = 0.5, x = 2, b = 1`. Forward: `w·x = 1`, `+ b = 2`,
`tanh(2) ≈ 0.964`. Now backward, right to left, seeding `n.grad = 1`:

```
n = tanh(s),  s = w·x + b = 2
  tanh rule:  s.grad = (1 − tanh(s)²)·n.grad = (1 − 0.964²)·1 ≈ 0.071
s = (w·x) + b                       a '+' passes the gradient straight through:
  (w·x).grad = 0.071                b.grad = 0.071
w·x = w · x                         a '*' gives each input the OTHER's value:
  w.grad = x·0.071 = 2·0.071  ≈ 0.142
  x.grad = w·0.071 = 0.5·0.071 ≈ 0.036
```

So `w.grad ≈ 0.142`: nudge this weight up and the neuron's output rises, gently. Every number came
from one of just three rules — tanh's `1 − tanh²`, addition's pass-through, multiplication's
swap — chained together. That is *exactly* what `loss.backward()` does across a whole MLP, only
with thousands of these little steps instead of three.

Now we stack neurons. A **`Layer`** is just a *row* of independent neurons, all seeing the same
inputs; an **`MLP`** (multi-layer perceptron) is a *stack* of layers, where each layer's outputs
become the next layer's inputs:

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]   # nout neurons, each taking nin inputs
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]                 # run every neuron on the same x
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts                                # e.g. [3, 4, 4, 1]
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)        # the output of one layer is the input to the next
        return x
```

So `MLP(3, [4, 4, 1])` reads "**3 inputs → a layer of 4 neurons → a layer of 4 → 1 output**." Count
the knobs: the first layer is 4 neurons × (3 weights + 1 bias) = 16; the second, 4 × (4+1) = 20;
the last, 1 × (4+1) = 5 — **41 `Value` parameters** in all, each a dial backprop can turn.

The one bit of glue is **`parameters()`**, which gathers every weight and bias from every neuron in
every layer into a single flat list:

```python
    def parameters(self):                                    # on the MLP
        return [p for layer in self.layers for p in layer.parameters()]   # all 41 Values
```

That flat list is exactly what the training loop steps over to update (`for p in
model.parameters(): p.data += ...`) and what `zero_grad()` steps over to reset. And nothing here is
new — it's `Value`s all the way down, so the *entire* MLP is automatically differentiable: one call
to `loss.backward()` and all 41 gradients appear at once.

---

## 9. Training: the same loop as Chapter 1, now fully transparent

Watch — this is the Chapter 1 loop (forward → backward → update), except `backward()`
is now **our** code:

```python
xs = [[2.0,3.0,-1.0], [3.0,-1.0,0.5], [0.5,1.0,1.0], [1.0,1.0,-1.0]]
ys = [1.0, -1.0, -1.0, 1.0]                  # the targets we want
model = MLP(3, [4, 4, 1])

for step in range(100):
    ypred = [model(x) for x in xs]                         # forward
    loss = sum((yp - yt)**2 for yt, yp in zip(ys, ypred))  # squared-error loss (a sum of squares)
    model.zero_grad()
    loss.backward()                                        # OUR engine fills every .grad
    for p in model.parameters():
        p.data += -0.05 * p.grad                           # update: step downhill
```

Every pass through that loop does four things, all now fully transparent to you:

1. **Forward** — run the model on the inputs; as a side effect this *builds the graph* of every
   `Value` from weights to predictions to loss.
2. **Loss** — collapse all the predictions into one number that says how wrong we are right now.
3. **Backward** — `loss.backward()` (your engine!) walks that graph in reverse and fills every
   `p.grad`: the sensitivity of the loss to each weight.
4. **Update** — nudge each weight a small step `−0.05 × grad`, i.e. *downhill* (opposite its
   gradient), exactly the "step opposite the slope" from the primer's valley.

(Why *squared* error for the loss? `(yp − yt)²` is `0` for a perfect prediction and grows as the
prediction drifts off — and squaring makes being wrong by `2` count *four* times as much as being
wrong by `1`, so the optimizer attacks the worst mistakes first. Summed over every example, it's a
single number that is small *exactly* when the model is right everywhere — which is why minimizing
it makes the model good.)

Repeat a hundred times and the loss walks down that valley. The `−0.05` is the **step size** (the
*learning rate*): too big and you leap past the bottom, too small and you crawl — a knob we tune
properly in Chapter 7.

Run [`code/train_demo.py`](./code/train_demo.py) and watch it learn:

```
step   0 | loss 6.005403
step  20 | loss 0.053542
step  50 | loss 0.012152
step  99 | loss 0.005104

targets:      [1.0, -1.0, -1.0, 1.0]
predictions:  [0.959, -0.992, -0.96, 0.958]
```

The loss falls a thousand-fold and the predictions snap to the targets. **You built the
autograd engine that made that happen.** No magic left.

---

## 10. This *is* PyTorch's autograd

Here's the punchline that ties the course together. Build the same expression in
PyTorch, call `.backward()`, and you get the **same gradients** your engine does:

```python
import torch
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([-3.0], requires_grad=True)
d = a * b + 10
d.backward()
print(a.grad, b.grad)     # tensor([-3.]) tensor([2.])  ← identical to our by-hand answer
```

Chapter 1's `loss.backward()` was doing precisely what you just built — recording a
graph on the forward pass, then walking it backward with the chain rule. The only
difference: PyTorch does it on whole **tensors** (arrays) at once instead of one scalar
at a time, which is what makes it fast enough for real models.

Why is that leap such a big deal? Our engine creates one `Value` — and one little `_backward`
closure — for *every single number*. A real model has billions of numbers, and billions of tiny
Python objects calling billions of tiny functions would be hopelessly slow. PyTorch instead wraps a
whole **array** of numbers in one object and applies each operation to *all* of them at once (on
hardware built for exactly that — Chapter 8). The *logic* is identical to what you just wrote — a
graph on the way forward, the chain rule on the way back — only the unit of work changes, from one
number to a whole array. We make that exact leap in Chapter 3.

> 🔎 **Verify it yourself.** In the exercises you'll run a full gradient-check: the same
> tangle of `+`, `*`, `**`, `tanh` in both your engine and PyTorch, asserting every
> gradient matches. (It does — to many decimal places.)

---

## 🤔 Common questions

- **Do I need to know calculus for this?** No. We used exactly one idea — "nudge the
  input, measure the output" — and a handful of local rules you can look up. That's
  enough to understand backprop completely.
- **Why scalars (one number at a time)? Isn't that slow?** Yes, hopelessly — micrograd
  is for *understanding*, not speed. Real engines batch everything into tensors
  (Chapter 3). The *logic* is identical; only the data type changes.
- **What's the difference between a derivative and a gradient?** A derivative is the
  sensitivity to *one* input. The "gradient" is just the collection of those
  derivatives for *all* the inputs (all the weights) at once.
- **Why did the loss stop at ~0.005 instead of 0?** It would keep creeping toward 0 with
  more steps; we stopped at 100. Tiny nonzero loss is fine — the predictions are already
  essentially right.
- **`_backward` stored as a function inside each object still feels weird.** It's a
  *closure*: a function that remembers the variables around it (here, `self`, `other`,
  `out`). That memory is exactly what lets it apply the right local rule later. Re-read
  §6 with that in mind and it clicks.
- **Why does the *order* of the backward pass matter so much?** A node can only hand its gradient
  to its parents once *its own* gradient is complete — i.e. after every node downstream of it has
  run. The reversed topological sort guarantees exactly that order. Run the nodes in the wrong
  order and a node would pass on a half-finished gradient, quietly corrupting everything upstream.
- **A `Value` is used in two places — does backprop double-count it?** It *sums*, which is the
  correct thing: each path contributes its gradient via `+=`, and the true sensitivity is the total
  over all paths (the `g = a*a` example in §6). The bug would be using `=` and keeping only the
  *last* path's contribution.

---

## ✅ Check your understanding

<details>
<summary>1. In one sentence, what is the gradient of the loss with respect to a weight?</summary>

How much the loss would change if you nudged that weight a tiny bit — its
*sensitivity*. Its sign and size tell gradient descent which way and how hard to push
that weight.
</details>

<details>
<summary>2. Backprop through <code>out = a + b</code>: if <code>out.grad</code> is 7, what do <code>a.grad</code> and <code>b.grad</code> become?</summary>

Both `+= 7`. Addition's local derivative is `1` for each input, so it passes the
incoming gradient straight through unchanged.
</details>

<details>
<summary>3. Backprop through <code>out = a * b</code> with <code>a = 4</code>, <code>b = 5</code>, <code>out.grad = 2</code>. What are the input grads?</summary>

Each input gets the *other* input's value times the incoming gradient:
`a.grad += b·out.grad = 5×2 = 10`, and `b.grad += a·out.grad = 4×2 = 8`.
</details>

<details>
<summary>4. Why do we accumulate with <code>+=</code> (and zero the grads before each backward pass)?</summary>

Because one Value can feed into several downstream computations; its true sensitivity is
the *sum* over all those paths. We add each path's contribution, so we must start from
0 each time (otherwise last step's gradients would leak into this one).
</details>

<details>
<summary>5. Why does a neural net need a nonlinearity like tanh?</summary>

Without one, every layer is a linear function, and stacking linear functions just gives
another linear function — so extra layers would add no power. The nonlinear squash is
what lets the network represent curved, complex relationships.
</details>

## 🎓 Key takeaways

- A **derivative** is a *sensitivity*: nudge an input, see how much the output moves.
  You can always measure one with a **finite difference**.
- A **`Value`** wraps a number and records how it was made, forming a **computational
  graph**.
- The **chain rule** says sensitivities *multiply* along a path; **backpropagation** is
  applying it from the output backward, reusing each node's gradient.
- Each operation has a simple **local rule** and stores a `_backward`; a **topological
  sort** + a reverse walk runs them in the right order. That's the whole engine.
- Stack `Value`-based **neurons** into an **MLP** and you can **train** it with the
  Chapter 1 loop — `loss.backward()` is now *your* code.
- This is exactly **PyTorch's autograd**, just scalar instead of tensor.

## 📖 New vocabulary

`derivative` · `gradient` · `finite difference` · `Value` · `computational graph` ·
`forward pass` · `backward pass` · `local derivative` · `chain rule` ·
`backpropagation` · `closure` · `topological sort` · `accumulating gradients` ·
`nonlinearity` · `tanh` · `neuron` · `layer` · `MLP` · `parameters`.

## 🧪 Practice & build

1. **The notebook** — [`code/explore.ipynb`](./code/explore.ipynb): measure a
   derivative, backprop by hand, build the engine, train a net. Do this first.
2. **Exercises** — [`exercises/`](./exercises/): extend the engine (`exp`, division),
   gradient-check against PyTorch, and a finite-difference checker. Tiered hints +
   solutions; a starter scaffold for the first one.
3. **Mini-project** — [`project/`](./project/): **"Curve Forge"**, where you use your
   engine to train a tiny network to trace a wiggly curve and *watch* it fit, from a
   starter (with a reference solution to check against).

## 🔗 Go deeper (optional)

- 🎥 Karpathy, [*The spelled-out intro to neural networks and backpropagation: building
  micrograd*](https://www.youtube.com/watch?v=VMj-3S1tku0) — ~2.5h; this chapter follows
  its path. The single best video on the subject.
- 💻 [karpathy/micrograd](https://github.com/karpathy/micrograd) — the original ~150-line
  repo our engine mirrors.
- 📄 [Christopher Olah, *Calculus on Computational Graphs: Backpropagation*](https://colah.github.io/posts/2015-08-Backprop/)
  — a beautifully illustrated explanation of *why* the backward walk is so efficient.

---

| ⬅️ Previous | Up | Next ➡️ |
|:--|:-:|--:|
| [Chapter 1 — Bigram](../01-bigram/) | [Syllabus](../../README.md#-syllabus) | [Chapter 3 — N-gram MLP](../03-ngram-mlp/) |
