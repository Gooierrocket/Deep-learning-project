import torch 
import torch.nn as nn
from torchdiffeq import odeint#_adjoint as odeint

from datetime import datetime

import definitions


t = definitions.t
X_train = definitions.X_train
X_test = definitions.X_test
dim = definitions.dim
func = definitions.ODEFunc(dim=dim)

opt = torch.optim.Adam(func.parameters(), lr=1e-3)
dt = (t[1] - t[0]).item()
win = 20                                  # steps per window
t_win = torch.arange(win) * dt

def get_batch(X, bs=128):
    n_steps, n_traj = X.shape[:2]
    i = torch.randint(0, n_steps - win, (bs,))
    j = torch.randint(0, n_traj, (bs,))
    x0 = X[i, j]                                                    # (bs, *state_shape)
    target = torch.stack([X[i + k, j] for k in range(win)])         # (win, bs, *state_shape)
    return x0, target


for it in range(2000):
    opt.zero_grad()
    x0, target = get_batch(X_train)
    pred = odeint(func, x0, t_win, method='dopri5', rtol=1e-4, atol=1e-6)
    loss = ((pred - target)**2).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(func.parameters(), 1.0)
    opt.step()
    if it % 200 == 0:
        print(it, loss.item(), "NFE:", func.nfe); func.nfe = 0


now = datetime.now() # current date and time
date_time = now.strftime("%m%d_%H%M")
torch.save(func, date_time + "_ode_func.pth")
