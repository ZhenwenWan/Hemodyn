import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import PhysicsModel as PM
import importlib
importlib.reload(PM)


class FeatureNN(nn.Module):
    def __init__(self, input_size, param_bounds=None):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        self.param_bounds = param_bounds

    def forward(self, x):
        out = self.extractor(x)
        if self.param_bounds is not None:
            lower, upper = map(torch.tensor, zip(*self.param_bounds))
            lower = lower.to(out.device)
            upper = upper.to(out.device)
            out = lower + (upper - lower) * out
        return out


def valid(alpha, velocity, t):
    Nx = 100
    dx = 1.0 / Nx
    model = PM.AdvDifAnalytical(nx=Nx)
    physics_layer = PM.PhysicsLayer(dx=dx, nx=Nx, nt=20)

    alpha    = torch.tensor(alpha, dtype=torch.float32)
    velocity = torch.tensor(velocity, dtype=torch.float32)
    t        = torch.tensor(t, dtype=torch.float32)
    u_analytical = model.analytical(alpha, velocity, t)
    u_physics = physics_layer.forward_solver(alpha, velocity, t)

    alpha    = torch.tensor(3.2e-5, dtype=torch.float32)
    velocity = torch.tensor(-0.0122, dtype=torch.float32)
    u_physics = physics_layer.forward_solver(alpha, velocity, t)

    loss_fn = nn.MSELoss()
    loss = loss_fn(u_analytical, u_physics)
    print(f"Loss {loss :.6f}")

    plt.figure(figsize=(6, 4))
    plt.plot(model.x.numpy(), u_analytical.detach().numpy(), label="Analytical")
    plt.plot(model.x.numpy(), u_physics.detach().numpy(), '--', label="Physics Layer")
    plt.title(f"Validation at alpha={alpha}, velocity={velocity}, t={t}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def prepare_training_data(model, true_alpha, true_velocity, num_samples, noise_std):
    inputs = []
    for i in range(num_samples):
        t = np.random.uniform(2.0, 11.0)
        #u_clean = model.analytical(true_alpha, true_velocity, t)

        alpha    = torch.tensor(true_alpha, dtype=torch.float32)
        velocity = torch.tensor(true_velocity, dtype=torch.float32)
        u_clean = model.forward_solver(alpha, velocity, t)

        u_noisy = u_clean + noise_std * torch.randn_like(u_clean)
        input_vec = torch.cat([u_noisy, torch.tensor([t])])
        inputs.append(input_vec)
    return torch.stack(inputs)


def main():
    Nx = 100
    dx = 1.0 / Nx
    model = PM.AdvDifAnalytical(nx=Nx)
    param_bounds = [(1.0e-5, 1.0e-4), (-1.0e-3, -2.0e-2)]

    num_samples = 100
    batch_size = 10
    input_size = Nx + 1 + 1  # u_noisy (Nx+1) + t
    nn_model = FeatureNN(input_size=input_size, param_bounds=param_bounds)
    physics_layer = PM.PhysicsLayer(dx=dx, nx=Nx, nt=20)

    for param in physics_layer.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    true_alpha = 4.0e-5
    true_velocity = -0.011
    noise_std = 0.01

    #training_data = prepare_training_data(model, true_alpha, true_velocity, num_samples, noise_std)
    training_data = prepare_training_data(physics_layer, true_alpha, true_velocity, num_samples, noise_std)

    epochs = 100
    loss_history = []
    alpha_history = []
    velocity_history = []

    for epoch in range(epochs):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        perm = torch.randperm(num_samples)
        training_data = training_data[perm]

        epoch_loss = 0.0
        alpha_vals = []
        velocity_vals = []

        for i in range(0, num_samples, batch_size):
            batch = training_data[i:i+batch_size]
            u_batch = batch[:, :-1]
            t_batch = batch[:, -1]

            optimizer.zero_grad()
            pred_params = nn_model(batch)
            alpha_batch = pred_params[:, 0]
            velocity_batch = pred_params[:, 1]

            losses = []
            for j in range(batch.shape[0]):
                output = physics_layer.forward_solver(alpha_batch[j], velocity_batch[j], t_batch[j])
                losses.append(loss_fn(u_batch[j], output))
            loss = torch.stack(losses).mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.shape[0]
            alpha_vals.extend(alpha_batch.tolist())
            velocity_vals.extend(velocity_batch.tolist())
            print(f"Sample {i+1}/{num_samples}, Loss {loss :.6f}, alpha_batch{alpha_batch.mean().item() :.9f}, velocity_batch{velocity_batch.mean().item() :.6f}")

        loss_history.append(epoch_loss / num_samples)
        alpha_history.append(np.mean(alpha_vals))
        velocity_history.append(np.mean(velocity_vals))
        scheduler.step(epoch_loss / num_samples)

        print(f"Epoch {epoch+1}/{epochs}, Avg Total Loss = {epoch_loss / num_samples:.6f}")

    pred_params = nn_model(training_data).detach()
    pred_alpha_mean = pred_params[:, 0].mean().item()
    pred_velocity_mean = pred_params[:, 1].mean().item()

    print("\n--- Recovered Parameters ---")
    print(f"True alpha = {true_alpha:.3f}, Predicted alpha (mean) = {pred_alpha_mean:.4f}")
    print(f"True velocity  = {true_velocity:.3f}, Predicted velocity  (mean) = {pred_velocity_mean:.4f}")

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
    plt.plot(velocity_history, label='Predicted β')
    plt.axhline(true_velocity, color='r', linestyle='--', label='True β')
    plt.title("Beta Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Mean β")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


main()

#valid(4.0e-5, -0.011, 10.2)

