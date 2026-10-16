import torch 
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn as nn

def linear_ODE(A, x, t):

    return x@A.T

A = torch.tensor(np.array([[-1, 10], [-10, -1]]), dtype=torch.float64)

f_linear = lambda t, x: linear_ODE(A, x, t)

# x_0 = torch.tensor(np.array([6, 7]), dtype=torch.float64)

# t = torch.tensor(np.linspace(0, 10, 100), dtype=torch.float64)

# data = odeint(f_linear, x_0, t)

torch.set_default_dtype(torch.float64)   # helps with tight tolerances and gradchecks
mu = 1.0

def true_f(t, x):                        # torchdiffeq expects f(t, x)
    return torch.stack([x[..., 1],
                        mu * (1 - x[..., 0]**2) * x[..., 1] - x[..., 0]], dim=-1)

def make_trajs(n, T=10.0, n_steps=201, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = (torch.rand(n, 2, generator=g) - 0.5) * 4          # x0 in [-2, 2]^2
    t = torch.linspace(0, T, n_steps)
    with torch.no_grad():
        X = odeint(f_linear, x0, t, rtol=1e-10, atol=1e-10, method='dopri5')  # (n_steps, n, 2)
    return t, X

t, X_train = make_trajs(64, seed=0)
_, X_test  = make_trajs(32, seed=1)   

class ODEFunc(nn.Module):
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        self.nfe = 0
    def forward(self, t, x):             # t unused, since the system is autonomous
        self.nfe += 1
        return self.net(x)

func = ODEFunc()

opt = torch.optim.Adam(func.parameters(), lr=1e-3)
dt = (t[1] - t[0]).item()
win = 20                                  # steps per window
t_win = torch.arange(win) * dt

def get_batch(X, bs=128):
    n_steps, n_traj, _ = X.shape
    i = torch.randint(0, n_steps - win, (bs,))
    j = torch.randint(0, n_traj, (bs,))
    x0 = X[i, j]                                                    # (bs, 2)
    target = torch.stack([X[i + k, j] for k in range(win)])         # (win, bs, 2)
    return x0, target

# X_train = data

for it in range(1000):
    opt.zero_grad()
    x0, target = get_batch(X_train)
    pred = odeint(func, x0, t_win, method='dopri5', rtol=1e-6, atol=1e-8)
    loss = ((pred - target)**2).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(func.parameters(), 1.0)
    opt.step()
    if it % 200 == 0:
        print(it, loss.item(), "NFE:", func.nfe); func.nfe = 0

# func.eval()

true_f = f_linear

@torch.no_grad()
def rollout(func, x0, t, **kw):
    return odeint(func, x0, t, rtol=1e-8, atol=1e-8, **kw)

pred = rollout(func, X_test[0], t)                 # full horizon T=10, much longer than the training window
err = (pred - X_test).norm(dim=-1)                 # (n_steps, n_traj)
print("mean error vs time:", err.mean(1)[::25])

g = torch.linspace(-3, 3, 40)
grid = torch.stack(torch.meshgrid(g, g, indexing='ij'), -1).reshape(-1, 2)
with torch.no_grad():
    rel = (func(0, grid) - true_f(0, grid)).norm(dim=-1) / (true_f(0, grid).norm(dim=-1) + 1e-8)
print("median rel. field error:", rel.median().item())

# plt.plot(t, data)
# plt.plot(data[:,0], data[:,1])
# plt.show()
