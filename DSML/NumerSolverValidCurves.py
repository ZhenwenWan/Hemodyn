import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def visualize_gradients(model):
    print("\n--- Gradient Info ---")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
        else:
            print(f"{name}: No gradient")


class Heat1DModelTorch:
    def __init__(self, nx=100):
        self.nx = nx
        self.x = torch.linspace(0, 1, nx + 1)

    def analytical(self, alpha, beta, t):
        alpha = torch.tensor(alpha, dtype=torch.float32)
        beta = torch.tensor(beta, dtype=torch.float32)
        t = torch.tensor(t, dtype=torch.float32)
        return (
            torch.exp(-np.pi**2 * alpha * t) * torch.sin(np.pi * self.x) +
            (beta / (np.pi**2 * alpha)) * (1 - torch.exp(-np.pi**2 * alpha * t)) * torch.sin(np.pi * self.x)
        )


class PhysicsLayer(nn.Module):
    def __init__(self, dx, nx, nt=20):
        super().__init__()
        self.dx = dx
        self.nx = nx
        self.nt = nt
        self.x = torch.linspace(0, 1, nx + 1)

    def forward_solver(self, alpha, beta, t):
        dt = t / self.nt
        r = alpha * dt / self.dx**2
        Nx = self.nx + 1

        u = torch.sin(np.pi * self.x)
        u[0] = 0
        u[-1] = 0
        u_internal = u[1:-1]

        main_diag = (1 + r) * torch.ones(Nx - 2)
        off_diag = (-r / 2) * torch.ones(Nx - 3)
        A = torch.diag(main_diag) + torch.diag(off_diag, -1) + torch.diag(off_diag, 1)

        B = torch.diag((1 - r) * torch.ones(Nx - 2))
        B += torch.diag((r / 2) * torch.ones(Nx - 3), -1)
        B += torch.diag((r / 2) * torch.ones(Nx - 3), 1)

        f = torch.sin(np.pi * self.x[1:-1])

        for _ in range(self.nt):
            rhs = torch.matmul(B, u_internal) + dt * beta * f
            u_internal = torch.linalg.solve(A, rhs)

        u_out = torch.zeros_like(u)
        u_out[1:-1] = u_internal
        return u_out


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
        raw = self.extractor(x)
        alpha = 0.1 + 0.9 * raw[:, 0:1]
        beta = 0.1 + 0.9 * raw[:, 1:2]
        return torch.cat([alpha, beta], dim=1)


def valid(alpha, beta, t):
    Nx = 100
    dx = 1.0 / Nx
    model = Heat1DModelTorch(nx=Nx)
    physics_layer = PhysicsLayer(dx=dx, nx=Nx)

    u_analytical = model.analytical(alpha, beta, t)
    u_physics = physics_layer.forward_solver(alpha, beta, t)

    plt.figure(figsize=(6, 4))
    plt.plot(model.x.numpy(), u_analytical.detach().numpy(), label="Analytical")
    plt.plot(model.x.numpy(), u_physics.detach().numpy(), '--', label="Physics Layer")
    plt.title(f"Validation at alpha={alpha}, beta={beta}, t={t}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    plt.close()


def main():
    Nx = 100
    dx = 1.0 / Nx
    model = Heat1DModelTorch(nx=Nx)
    nn_model = FeatureNN(input_size=Nx + 1)
    physics_layer = PhysicsLayer(dx=dx, nx=Nx)

    for param in physics_layer.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    true_alpha = 0.75
    true_beta = 0.63
    num_samples = 100
    noise_std = 0.01

    inputs = []
    times = []
    for _ in range(num_samples):
        t = np.random.uniform(0.01, 1.0)
        times.append(t)
        u_clean = model.analytical(true_alpha, true_beta, t)
        u_noisy = u_clean + noise_std * torch.randn_like(u_clean)
        inputs.append(u_noisy)

    true_inputs = torch.stack(inputs)
    times = torch.tensor(times)

    epochs = 200
    loss_history = []
    alpha_history = []
    beta_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        alpha_vals = []
        beta_vals = []
        for i in range(num_samples):
            optimizer.zero_grad()
            input_i = true_inputs[i].unsqueeze(0)
            t_i = times[i]

            pred_params = nn_model(input_i)
            alpha_i = pred_params[0, 0]
            beta_i = pred_params[0, 1]
            output = physics_layer.forward_solver(alpha_i, beta_i, t_i)

            loss = loss_fn(input_i.squeeze(), output)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            alpha_vals.append(alpha_i.item())
            beta_vals.append(beta_i.item())

        loss_history.append(epoch_loss / num_samples)
        alpha_history.append(np.mean(alpha_vals))
        beta_history.append(np.mean(beta_vals))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Avg Total Loss = {epoch_loss / num_samples:.6f}")

    pred_params = nn_model(true_inputs).detach()
    pred_alpha_mean = pred_params[:, 0].mean().item()
    pred_beta_mean = pred_params[:, 1].mean().item()

    print("\n--- Recovered Parameters ---")
    print(f"True alpha = {true_alpha:.3f}, Predicted alpha (mean) = {pred_alpha_mean:.4f}")
    print(f"True beta  = {true_beta:.3f}, Predicted beta  (mean) = {pred_beta_mean:.4f}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(alpha_history, label='Predicted α')
    plt.axhline(true_alpha, color='r', linestyle='--', label='True α')
    plt.title("Alpha Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Mean α")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(beta_history, label='Predicted β')
    plt.axhline(true_beta, color='r', linestyle='--', label='True β')
    plt.title("Beta Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Mean β")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

