import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class Heat1DModel:
    def __init__(self, nx=100, nt=100, dx=0.01, dt=0.01):
        self.nx = nx
        self.nt = nt
        self.dx = dx
        self.dt = dt
        self.x = np.linspace(0, 1, nx + 1)

    def analytical(self, alpha, beta, t):
        return np.exp(-np.pi**2 * alpha * t) * np.sin(np.pi * self.x) + \
               (beta / (np.pi**2 * alpha)) * (1 - np.exp(-np.pi**2 * alpha * t)) * np.sin(np.pi * self.x)

    def numerical(self, alpha, beta, t):
        if t <= 0 or t > 1:
            raise ValueError("t must be in the range (0, 1]")
        dt = t / 100
        Nt = 100
        Nx = self.nx + 1
        dx = 1.0 / self.nx
        x = np.linspace(0, 1, Nx)
        u = np.sin(np.pi * x)
        u[0] = u[-1] = 0

        r = alpha * dt / dx**2
        main_diag = (1 + r) * np.ones(Nx - 2)
        off_diag = (-r / 2) * np.ones(Nx - 3)
        A = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csc')

        B = sp.diags([r / 2 * np.ones(Nx - 3), (1 - r) * np.ones(Nx - 2), r / 2 * np.ones(Nx - 3)],
                     offsets=[-1, 0, 1], format='csc')

        u_internal = u[1:-1].copy()
        for _ in range(Nt):
            rhs = B.dot(u_internal) + dt * beta * np.sin(np.pi * x[1:-1])
            u_internal = spla.spsolve(A, rhs)

        u[1:-1] = u_internal
        return u


class FeatureNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.extractor(x)


def main():
    Nx = 100
    model = Heat1DModel(nx=Nx)
    nn_model = FeatureNN(input_size=Nx + 1)
    optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Known ground truth parameters
    true_alpha = 0.5
    true_beta = 0.3
    t = 1.0
    num_samples = 100

    # Generate identical samples from analytical model
    true_inputs_np = np.array([model.analytical(true_alpha, true_beta, t) for _ in range(num_samples)])
    true_inputs = torch.tensor(true_inputs_np, dtype=torch.float32)

    # Create matching true targets for training
    true_targets = torch.tensor([[true_alpha, true_beta]] * num_samples, dtype=torch.float32)

    epochs = 1000
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_params = nn_model(true_inputs)
        loss = loss_fn(pred_params, true_targets)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss = {loss.item():.6f}")

    # Evaluation: use predicted alpha/beta in numerical solver
    pred_params = nn_model(true_inputs).detach()
    pred_alpha_mean = pred_params[:, 0].mean().item()
    pred_beta_mean = pred_params[:, 1].mean().item()

    print("\n--- Recovered Parameters ---")
    print(f"True alpha = {true_alpha:.3f}, Predicted alpha (mean) = {pred_alpha_mean:.4f}")
    print(f"True beta  = {true_beta:.3f}, Predicted beta  (mean) = {pred_beta_mean:.4f}")

    # Use predicted mean to run numerical model for comparison
    u_numerical = model.numerical(pred_alpha_mean, pred_beta_mean, t)
    u_true = model.analytical(true_alpha, true_beta, t)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(model.x, u_true, label='Analytical Input')
    plt.plot(model.x, u_numerical, label='Numerical from Predicted', linestyle='--')
    plt.legend()
    plt.title("Model Evaluation")

    plt.subplot(2, 1, 2)
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


main()

