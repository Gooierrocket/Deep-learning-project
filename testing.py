import torch 
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchdiffeq import odeint

import definitions

### define the true ODE function and the learned ODE function
f_linear = definitions.f_linear
X_test = definitions.X_test
t = definitions.t

func = torch.load('1017_2127_ode_func.pth', weights_only=False)
func.eval()

true_f = f_linear


### define a rollout function to evaluate the learned ODE function
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


### plotting the vector field
fig, ax = plt.subplots(2, 2)

with torch.no_grad():
    true_vec = true_f(0, grid)
    pred_vec = func(0, grid)

ax[0, 0].quiver(
    grid[:, 0].numpy(), grid[:, 1].numpy(),
    true_vec[:, 0].numpy(), true_vec[:, 1].numpy(),
    color="blue"
); ax[0, 0].set_title("True vector field")

ax[0, 1].quiver(
    grid[:, 0].numpy(), grid[:, 1].numpy(),
    pred_vec[:, 0].numpy(), pred_vec[:, 1].numpy(),
    color="red"
); ax[0, 1].set_title("Learned vector field")

ax[1, 1].quiver(
    grid[:, 0].numpy(), grid[:, 1].numpy(),
    (pred_vec[:, 0].numpy() - true_vec[:, 0].numpy()) / (abs(true_vec[:, 0].numpy()) + 1),
    (pred_vec[:, 1].numpy() - true_vec[:, 1].numpy()) / (abs(true_vec[:, 1].numpy()) + 1),
    color="orange"
); ax[1, 1].set_title("Difference vector field")


ax[1, 0].quiver(
    grid[:, 0].numpy(), grid[:, 1].numpy(),
    true_vec[:, 0].numpy(), true_vec[:, 1].numpy(),
    color="blue", label="True field")
ax[1, 0].quiver(
    grid[:, 0].numpy(), grid[:, 1].numpy(),
    pred_vec[:, 0].numpy(), pred_vec[:, 1].numpy(),
    color="red", label="Learned field")
ax[1, 0].set_title("True vs Learned vector field")
ax[1, 0].legend()

plt.show()