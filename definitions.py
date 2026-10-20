import torch 
import numpy as np
import torch.nn as nn
from torchdiffeq import odeint#_adjoint as odeint
import torch.nn as nn

torch.set_default_dtype(torch.float64)   # helps with tight tolerances and gradchecks

def linear_ODE(A, x, t):

    return x@A.T

A = torch.tensor(np.array([[-1, 10], [-10, -1]]), dtype=torch.float64)

f_linear = lambda t, x: linear_ODE(A, x, t)

noise = True
mu = 1.0

def Van_der_Pol_f(t, x):                        # torchdiffeq expects f(t, x)
    return torch.stack([x[..., 1],
                        mu * (1 - x[..., 0]**2) * x[..., 1] - x[..., 0]], dim=-1)

def lorentz_f(t, x):
    return torch.stack([10 * (x[..., 1] - x[..., 0]),
                        x[..., 0] * (28 - x[..., 2]) - x[..., 1],
                        x[..., 0] * x[..., 1] - 8/3 * x[..., 2]], dim=-1)

true_f = Van_der_Pol_f
dim = 2

def make_trajs(n, dim=2, limit=4, T=10.0, n_steps=201, noise=False, noise_power = 0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = (torch.rand(n, dim, generator=g) - 0.5) * 2*limit          # x0 in [-limit, limit]^dim
    t = torch.linspace(0, T, n_steps)
    with torch.no_grad():
        X = odeint(true_f, x0, t, rtol=1e-10, atol=1e-10, method='dopri5')  # (n_steps, n, 2)
    if noise:
        X_noisy = X + noise_power * torch.randn_like(X, generator=g)
        return t, X, X_noisy
    else:
        return t, X, None

t, X_clean, X_train = make_trajs(248, dim=dim, seed=0, limit=4, noise=noise)
_, X_test, _  = make_trajs(32, dim=dim, seed=1)   

class ODEFunc(nn.Module):
    def __init__(self, dim=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),

            nn.Linear(hidden, hidden*16), nn.Tanh(),

            nn.Linear(hidden*16, hidden), nn.Tanh(),

            nn.Linear(hidden, dim),
        )
        self.nfe = 0
    def forward(self, t, x):             # t unused, since the system is autonomous
        self.nfe += 1
        return self.net(x)