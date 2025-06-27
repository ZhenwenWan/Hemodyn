import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class AdvDifAnalytical:
    def __init__(self, nx=400):
        self.nx = nx
        self.x = torch.linspace(0, 1, nx + 1)

    def analytical(self, alpha, velocity, t):
        alpha    = torch.tensor(alpha, dtype=torch.float32)
        velocity = torch.tensor(velocity, dtype=torch.float32)
        t        = torch.tensor(t, dtype=torch.float32)
        pi = np.pi
        x_shift = self.x.to(alpha.device if isinstance(alpha, torch.Tensor) else None) - velocity * t
        return (1 / torch.sqrt(4 * pi * alpha * t)) * torch.exp(-((x_shift - 0.5)**2) / (4 * alpha * t))


class ForwardRHS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, velocity, physics, t):
        ctx.save_for_backward(alpha, velocity)
        ctx.physics = physics
        ctx.t = t
        with torch.no_grad():
            result = physics._forward_simulation(alpha, velocity, t)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        alpha, velocity = ctx.saved_tensors
        physics = ctx.physics
        t = ctx.t
        one = 1.0
        eps = 1.0e-3

        with torch.no_grad():
            u_base = physics._forward_simulation(alpha, velocity, t)

            u_alpha = physics._forward_simulation(alpha*(one + eps), velocity, t)
            grad_alpha = ((u_alpha - u_base) /(alpha*eps)) @ grad_output

            u_velocity = physics._forward_simulation(alpha, velocity*(one + eps), t)
            grad_velocity = ((u_velocity - u_base) /(velocity*eps)) @ grad_output

        return grad_alpha, grad_velocity, None, None

class PhysicsLayer(nn.Module):

    def __init__(self, dx, nx, nt=100):
        super().__init__()
        self.dx = dx
        self.nx = nx
        self.nt = nt
        self.x = torch.linspace(0, 1, nx + 1)

    def implicit_step(self, u, dt, alpha, velocity, inlet):
        dx = self.dx
        Nx = self.nx + 1
        device = u.device

        r_diff = alpha / dx**2
        r_adv = velocity / (2 * dx)

        A = torch.zeros((Nx-1, Nx-1), device=device)

        for i in range(1, Nx - 2):
            A[i, i - 1] = r_diff + r_adv
            A[i, i]     = - 2 * r_diff - 1/dt
            A[i, i + 1] = r_diff - r_adv

        A[0, 0] = 1.0
        A[-1, -2] =   r_diff + r_adv 
        A[-1, -1] = - r_diff - r_adv

        rhs_vec = - u.clone()/dt
        rhs_vec[0] = inlet

        u_new = torch.zeros(Nx, device=device)
        u_new[:-1] = torch.linalg.solve(A, rhs_vec[:-1])
        u_new[-1] = u_new[-2]
        return u_new

    def forward_solver(self, alpha, velocity, t):
        return ForwardRHS.apply(alpha, velocity, self, t)

    def _forward_simulation(self, alpha, velocity, t):
        dt = (t - 1) / self.nt
        dx = self.dx
        Nx = self.nx + 1

        alpha = alpha.float()
        velocity = velocity.float()
        device = alpha.device

        pi = np.pi
        x = self.x.to(device)

        t0 = torch.tensor(1.0, dtype=torch.float32, device=device)
        u0 = (1 / torch.sqrt(4 * pi * alpha * t0)) * torch.exp(-((x - 0.5 - velocity * t0)**2) / (4 * alpha * t0))
        u1 = u0.clone()

        for n in range(self.nt - 1):
            tn = t0 + dt * (n + 1)
            tn_tensor = tn.float() if isinstance(tn, torch.Tensor) else torch.tensor(tn, dtype=torch.float32, device=device)
            tn1_tensor = tn_tensor + dt

            inlet_n = (1 / torch.sqrt(4 * pi * alpha * tn_tensor)) * torch.exp(-((0.0 - 0.5 - velocity * tn_tensor)**2) / (4 * alpha * tn_tensor))
            inlet_n1 = (1 / torch.sqrt(4 * pi * alpha * tn1_tensor)) * torch.exp(-((0.0 - 0.5 - velocity * tn1_tensor)**2) / (4 * alpha * tn1_tensor))

            u1 = self.implicit_step(u0, dt, alpha, velocity, inlet_n1)

            u0 = u1

        return u1
