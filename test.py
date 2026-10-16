import torch 
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn as nn

import ODE_project

func = ODE_project.ODEFunc()
func.load_state_dict(torch.load("ode_func.pth"))
t, _ = ODE_project.make_trajs(64, seed=0)
_, X_test  = ODE_project.make_trajs(32, seed=1) 

def linear_ODE(A, x, t):

    return x@A.T

A = torch.tensor(np.array([[-1, 10], [-10, -1]]), dtype=torch.float64)

f_linear = lambda t, x: linear_ODE(A, x, t)

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