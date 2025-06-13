import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

class Heat1DModel:
    def __init__(self, alpha=0.1, beta=0.1, nx=100):
        self.alpha = alpha
        self.beta = beta
        self.nx = nx
        self.x = np.linspace(0, 1, nx + 1)
        self.dx = self.x[1] - self.x[0]
        self.u0 = np.sin(np.pi * self.x)
        self.u0[0] = self.u0[-1] = 0

    def analytical(self, t):
        term1 = np.exp(-np.pi**2 * self.alpha * t) * np.sin(np.pi * self.x)
        term2 = self.beta / (np.pi**2 * self.alpha) * (1 - np.exp(-np.pi**2 * self.alpha * t)) * np.sin(np.pi * self.x)
        return term1 + term2

    def numerical(self, t):
        if t <= 0 or t > 1:
            raise ValueError("t must be in the range (0, 1]")
        dt = t / 100
        Nt = 100
        u = self.u0.copy()
        Nx = self.nx + 1

        r = self.alpha * dt / self.dx**2
        main_diag = (1 + r) * np.ones(Nx - 2)
        off_diag = (-r / 2) * np.ones(Nx - 3)
        A = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csc')

        B = sp.diags([r / 2 * np.ones(Nx - 3), (1 - r) * np.ones(Nx - 2), r / 2 * np.ones(Nx - 3)],
                     offsets=[-1, 0, 1], format='csc')

        u_internal = u[1:-1].copy()
        for _ in range(Nt):
            rhs = B.dot(u_internal) + dt * self.beta * np.sin(np.pi * self.x[1:-1])
            u_internal = spla.spsolve(A, rhs)
        u[1:-1] = u_internal
        u[0] = u[-1] = 0
        return u

# Repeat the plots for four combinations
alphas_betas = [(0.1, 0.1), (0.1, 0.5), (0.5, 0.1), (0.5, 0.5)]
t_query = 0.5

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, (alpha, beta) in enumerate(alphas_betas):
    model = Heat1DModel(alpha=alpha, beta=beta)
    u_ana = model.analytical(t_query)
    u_num = model.numerical(t_query)
    
    axes[i].plot(model.x, u_ana, label='Analytical', lw=2)
    axes[i].plot(model.x, u_num, '--', label='Numerical', lw=2)
    axes[i].set_title(f'α={alpha}, β={beta}')
    axes[i].legend()
    axes[i].grid(True)

plt.suptitle("Comparison of Analytical vs Numerical Solutions")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

