import torch 
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchdiffeq import odeint
from scipy.optimize import root

import definitions

### define the true ODE function and the learned ODE function
f_linear = definitions.f_linear
X_test = definitions.X_test
t = definitions.t
dim = definitions.dim
noise = definitions.noise

# func = torch.load('1017_2127_ode_func.pth', weights_only=False)
# func = torch.load('1018_1300_ode_func.pth', weights_only=False)
# func = torch.load('1018_1801_ode_func.pth', weights_only=False)
# func = torch.load('1019_1758_ode_func.pth', weights_only=False)
func = torch.load('1020_1956_ode_func.pth', weights_only=False)

func.eval()

true_f = definitions.true_f

### define a rollout function to evaluate the learned ODE function
@torch.no_grad()
def rollout(func, x0, t, **kw):
    return odeint(func, x0, t, rtol=1e-8, atol=1e-8, **kw)

pred = rollout(func, X_test[0], t)                 # full horizon T=10, much longer than the training window
err = (pred - X_test).norm(dim=-1)                 # (n_steps, n_traj)
print("mean error vs time:", err.mean(1)[::25])

g = torch.linspace(-3, 3, 40)

# [axis_coords] * dim

grid = torch.stack(torch.meshgrid(*[g] * dim, indexing='ij'), -1).reshape(-1, dim)
with torch.no_grad():
    rel = (func(0, grid) - true_f(0, grid)).norm(dim=-1) / (true_f(0, grid).norm(dim=-1) + 1e-8)
print("median rel. field error:", rel.median().item())

# origin = torch.tensor([0, 0], dtype=torch.float64)

func_without_t = lambda x: func(0, x)

def objective_function(x_numpy):
    x_tensor = torch.tensor(x_numpy, dtype=torch.float64).reshape(1, dim)
    
    with torch.no_grad():
        output = func_without_t(x_tensor)
        
    return output.numpy().flatten()

initial_guess = torch.randn(dim, dtype=torch.float64).numpy()

sol = root(objective_function, initial_guess, method='lm')

fixed_points = torch.tensor(sol.x)
jac = torch.autograd.functional.jacobian(func_without_t, fixed_points)

print(f'Fixed point(s): {fixed_points}')
print(f'Jacobian at fixed point(s): {jac}')
print(f'Eigenvalues of Jacobian at fixed point(s): {torch.linalg.eigvals(jac)}')

### plotting the vector field
if dim == 2:
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

    true_pred = rollout(true_f, X_test[0], t)
    fig, ax = plt.subplots()

    for i in range(X_test.shape[1]):
        ax.plot(true_pred[:, i, 0].numpy(), true_pred[:, i, 1].numpy(),
            color="blue", alpha=0.7,
            label="True system" if i == 0 else None)
        ax.plot(pred[:, i, 0].numpy(), pred[:, i, 1].numpy(),
            color="red", linestyle="--", alpha=0.7,
            label="Neural ODE" if i == 0 else None)

    ax.scatter(
        X_test[0, :, 0].numpy(), X_test[0, :, 1].numpy(),
        color="black", s=20, marker="o", label="Starting points"
    )

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("True system vs Neural ODE trajectories")
    ax.legend()
    ax.grid(True)
    plt.show()

elif dim == 3:
    true_pred = rollout(true_f, X_test[0], t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(X_test.shape[1]):
        ax.plot(true_pred[:, i, 0].numpy(), true_pred[:, i, 1].numpy(),
            true_pred[:, i, 2].numpy(), color="blue", alpha=0.7,
            label="True system" if i == 0 else None)
        ax.plot(pred[:, i, 0].numpy(), pred[:, i, 1].numpy(),
            pred[:, i, 2].numpy(), color="red", linestyle="--", alpha=0.7,
            label="Neural ODE" if i == 0 else None)

    ax.scatter(
        X_test[0, :, 0].numpy(), X_test[0, :, 1].numpy(), X_test[0, :, 2].numpy(),
        color="black", s=20, marker="o", label="Starting points"
    )

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("x₃")
    ax.set_title("True system vs Neural ODE trajectories")
    ax.legend()
    ax.grid(True)
    plt.show()

if noise:
    # Compare a few noisy training trajectories with Neural ODE rollouts
    # initialized from the same observed starting points.
    train_data = definitions.X_train
    clean_data = definitions.X_clean
    n_train = min(5, train_data.shape[1])
    scatter_stride = 5  # Plot every nth noisy observation to reduce clutter.
    train_pred = rollout(func, train_data[0, :n_train], t)

    if dim == 2:
        fig, ax = plt.subplots()
        for i in range(n_train):
            points = ax.scatter(
                train_data[::scatter_stride, i, 0].numpy(),
                train_data[::scatter_stride, i, 1].numpy(),
                alpha=0.7, label=f"Noisy training data {i + 1}"
            )
            ax.plot(
                clean_data[:, i, 0].numpy(), clean_data[:, i, 1].numpy(),
                color=points.get_facecolor()[0], linestyle="-",
                label=f"Clean training trajectory {i + 1}"
            )
            ax.plot(
                train_pred[:, i, 0].numpy(), train_pred[:, i, 1].numpy(),
                color=points.get_facecolor()[0], linestyle="--",
                label=f"Neural ODE prediction {i + 1}"
            )
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_title("Noisy training trajectories vs Neural ODE predictions")
        ax.legend()
        ax.grid(True)
        plt.show()

    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i in range(n_train):
            points = ax.scatter(
                train_data[::scatter_stride, i, 0].numpy(),
                train_data[::scatter_stride, i, 1].numpy(),
                train_data[::scatter_stride, i, 2].numpy(), alpha=0.7,
                label=f"Noisy training data {i + 1}"
            )
            ax.plot(
                clean_data[:, i, 0].numpy(), clean_data[:, i, 1].numpy(),
                clean_data[:, i, 2].numpy(), color=points.get_facecolor()[0],
                linestyle="-", label=f"Clean training trajectory {i + 1}"
            )
            ax.plot(
                train_pred[:, i, 0].numpy(), train_pred[:, i, 1].numpy(),
                train_pred[:, i, 2].numpy(), color=points.get_facecolor()[0],
                linestyle="--", label=f"Neural ODE prediction {i + 1}"
            )
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_zlabel("x₃")
        ax.set_title("Noisy training trajectories vs Neural ODE predictions")
        ax.legend()
        ax.grid(True)
        plt.show()