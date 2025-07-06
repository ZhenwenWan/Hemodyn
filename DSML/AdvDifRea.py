import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class GradiSurr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, velocity, r0, r1, physics, t):
        ctx.save_for_backward(alpha, velocity, r0, r1)
        ctx.physics = physics
        ctx.t = t
        with torch.no_grad():
            result = physics._forward_simulation(alpha, velocity, r0, r1, t)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        alpha, velocity, r0, r1 = ctx.saved_tensors
        physics = ctx.physics
        t = ctx.t
        one = 1.0
        eps = 1.0e-3

        with torch.no_grad():
            u_base = physics._forward_simulation(alpha, velocity, r0, r1, t)

            u_alpha = physics._forward_simulation(alpha * (one + eps), velocity, r0, r1, t)
            grad_alpha = torch.sum((u_alpha - u_base) * grad_output) / (alpha * eps)
            u_velocity = physics._forward_simulation(alpha, velocity * (one + eps), r0, r1, t)
            grad_velocity = torch.sum((u_velocity - u_base) * grad_output) / (velocity * eps)
            u_r0 = physics._forward_simulation(alpha, velocity, r0 * (one + eps), r1, t)
            grad_r0 = torch.sum((u_r0 - u_base) * grad_output) / (r0 * eps)

            u_r1 = physics._forward_simulation(alpha, velocity, r0, r1 * (one + eps), t)
            grad_r1 = torch.sum((u_r1 - u_base) * grad_output) / (r1 * eps)

        return grad_alpha, grad_velocity, grad_r0, grad_r1, None, None


class PhysicsLayer(nn.Module):

    def __init__(self, dx=0.01, nx=100, dt=0.01, nt=100, t0=0.0, x0=0.0):
        super().__init__()
        self.dx = dx
        self.nx = nx
        self.dt = dt
        self.nt = nt
        self.t0 = t0
        self.x  = torch.linspace(x0, x0 + nx*dx, nx + 1)

    def initial_condition(self, alpha, velocity):
        x = self.x.to(alpha.device)
        return torch.full_like(x, 1.0)

    def boundary_condition(self, alpha, velocity, t):
        pi = np.pi
        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=alpha.device)
        bc = 1.0 + 0.5 * torch.sin(2 * pi * t_tensor)
        return bc

    def assemble_matrix(self, alpha, velocity, r1, dt):
        dx = self.dx
        Nx = self.nx + 1
        device = alpha.device

        r_diff = alpha / dx**2
        r_adv = velocity / (2 * dx)
        A = torch.zeros((Nx-1, Nx-1), device=device)

        for i in range(1, Nx - 2):
            A[i, i - 1] = r_diff + r_adv
            A[i, i]     = - 2 * r_diff - 1/dt - r1
            A[i, i + 1] = r_diff - r_adv

        A[0,   0] = 1.0
        A[-1, -2] =    r_diff + r_adv
        A[-1, -1] = - (r_diff + r_adv) - 1/dt - r1

        return A

    def solve_linear_system(self, A, rhs):
        u_new = torch.zeros_like(self.x, device=rhs.device)
        u_new[:-1] = torch.linalg.solve(A, rhs[:-1])
        u_new[-1] = u_new[-2]
        return u_new

    def time_integration(self, alpha, velocity, r0, r1, t_batch):
        alpha = alpha.float()
        velocity = velocity.float()
        r0 = r0.float()
        r1 = r1.float()
        device = alpha.device
        x = self.x.to(alpha.device)

        if isinstance(t_batch, list):
            t_batch = torch.stack(t_batch)

        t_min = self.t0
        if t_batch.min().item() < self.t0:
            raise ValueError("Input error: t_batch contains times earlier than t0.")
        t_max = t_batch.max().item()
        dt = (t_max - t_min) / self.nt
        self.dt = dt

        u = self.initial_condition(alpha, velocity)
        time_points = torch.linspace(t_min, t_max, self.nt + 1, device=device)

        A = self.assemble_matrix(alpha, velocity, r1, dt)

        all_solutions = []
        all_solutions.append(u.clone())

        for n in range(self.nt):
            tn1 = t_min + (n + 1) * dt
            inlet_n1 = self.boundary_condition(alpha, velocity, tn1)
            reaction_term = r0*x
            rhs_vec = (- u.clone() / dt + reaction_term)
            rhs_vec[0] = inlet_n1

            u = self.solve_linear_system(A, rhs_vec)
            all_solutions.append(u.clone())

        U_all = torch.stack(all_solutions)  # Shape: [nt+1, nx+1]

        # Interpolate to each t in t_batch
        t_norm = (t_batch - t_min) / (t_max - t_min) * self.nt
        t_idx0 = torch.clamp(t_norm.floor().long(), 0, self.nt - 1)
        t_idx1 = t_idx0 + 1
        w = (t_norm - t_idx0.float()).unsqueeze(1)  # interpolation weight

        u_interp = (1 - w) * U_all[t_idx0] + w * U_all[t_idx1]
        return u_interp

    def forward_solver(self, alpha, velocity, r0, r1, t):
        return GradiSurr.apply(alpha, velocity, r0, r1, self, t)

    def _forward_simulation(self, alpha, velocity, r0, r1, t):
        return self.time_integration(alpha, velocity, r0, r1, t)

