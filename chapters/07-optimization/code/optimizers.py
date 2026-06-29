"""
Optimizers from scratch: SGD, Momentum, Adam, AdamW.
====================================================
Every chapter so far called `torch.optim.AdamW` and trusted it. Here we open it up. An optimizer
takes the parameters and their gradients and decides *how to step*. We implement four, from the
plain to the modern default, and race them on a small network.

Each `.step(params, grads)` updates `params` in place (no autograd — we hand it the grads).

Run:  python optimizers.py
"""
import math
import torch


class SGD:
    """Plain gradient descent: step downhill, fixed size."""

    def __init__(self, lr=0.1):
        self.lr = lr

    @torch.no_grad()
    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Momentum:
    """SGD with velocity: accumulate a running average of gradients and roll."""

    def __init__(self, lr=0.1, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    @torch.no_grad()
    def step(self, params, grads):
        if self.v is None:
            self.v = [torch.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.v[i].mul_(self.beta).add_(g)        # v = beta*v + g
            p -= self.lr * self.v[i]


class Adam:
    """Per-parameter adaptive step sizes (RMSprop) + momentum, with bias correction."""

    def __init__(self, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    @torch.no_grad()
    def step(self, params, grads):
        if self.m is None:
            self.m = [torch.zeros_like(p) for p in params]
            self.v = [torch.zeros_like(p) for p in params]
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g          # 1st moment (mean)
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g      # 2nd moment (variance)
            mhat = self.m[i] / (1 - self.b1 ** self.t)                   # bias-correct
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (vhat.sqrt() + self.eps)


class AdamW(Adam):
    """Adam + decoupled weight decay: gently pull weights toward zero each step."""

    def __init__(self, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-2):
        super().__init__(lr, beta1, beta2, eps)
        self.wd = weight_decay

    @torch.no_grad()
    def step(self, params, grads):
        for p in params:
            p -= self.lr * self.wd * p              # decoupled decay (separate from the gradient)
        super().step(params, grads)


# ----------------------------------------------------------------------------------------------
# A small task to race them on: classify two interleaving half-moons.
# ----------------------------------------------------------------------------------------------
def make_moons(n=400, noise=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n // 2, generator=g) * math.pi
    top = torch.stack([torch.cos(t), torch.sin(t)], 1)
    bot = torch.stack([1 - torch.cos(t), 0.5 - torch.sin(t)], 1)
    X = torch.cat([top, bot]) + noise * torch.randn(n, 2, generator=g)
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
    return X, y


def fresh_model(seed=1337):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(2, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 2),
    )


def train_with(opt, steps=300):
    X, y = make_moons()
    model = fresh_model()
    params = list(model.parameters())
    losses = []
    for _ in range(steps):
        logits = model(X)
        loss = torch.nn.functional.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()
        opt.step(params, [p.grad for p in params])
        losses.append(loss.item())
    return losses


def verify_against_torch():
    """Our Adam should make the same updates as torch.optim.Adam."""
    torch.manual_seed(0)
    p_ours = torch.randn(5)
    p_torch = p_ours.clone()
    ours = Adam(lr=0.1)
    topt = torch.optim.Adam([p_torch], lr=0.1)
    for _ in range(20):
        g = 2 * p_torch                      # gradient of f = sum(p^2)
        ours.step([p_ours], [2 * p_ours])
        p_torch.grad = g
        topt.step()
    return torch.allclose(p_ours, p_torch, atol=1e-5)


def main():
    print("our Adam matches torch.optim.Adam:", verify_against_torch())
    print("\nracing optimizers on two-moons (final loss after 300 steps):")
    for name, opt in [("SGD      ", SGD(lr=0.5)),
                      ("Momentum ", Momentum(lr=0.1, beta=0.9)),
                      ("Adam     ", Adam(lr=0.05)),
                      ("AdamW    ", AdamW(lr=0.05, weight_decay=0.01))]:
        losses = train_with(opt)
        print(f"  {name} start {losses[0]:.3f} -> final {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
