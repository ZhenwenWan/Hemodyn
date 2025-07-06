import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import AdvDifRea as PM
import importlib
importlib.reload(PM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureNN(nn.Module):
    def __init__(self, input_size, param_bounds=None):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
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

def prepare_training_data(model, true_alpha, true_velocity, true_r0, true_r1, num_samples, noise_std):
    time_vec = []
    for i in range(num_samples):
        t = np.random.uniform(0.0, 1.0)
        time_vec.append(torch.tensor(t, dtype=torch.float32))

    alpha    = torch.tensor(true_alpha, dtype=torch.float32)
    velocity = torch.tensor(true_velocity, dtype=torch.float32)
    r0       = torch.tensor(true_r0, dtype=torch.float32)
    r1       = torch.tensor(true_r1, dtype=torch.float32)
    u_clean  = model.forward_solver(alpha, velocity, r0, r1, time_vec)
    u_noisy  = u_clean + noise_std * torch.randn_like(u_clean)

    inputs = []
    for i in range(num_samples):
        input_vec = torch.cat([u_noisy[i], time_vec[i].unsqueeze(0)])
        inputs.append(input_vec)
    return torch.stack(inputs)

def main():
    Nx = 100
    dx = 1.0 / Nx
    param_bounds = [(1.0e-4, 1.0e-2), (1.0e-3, 5.0e-2), (1.0e-2, 2.0e-1), (1.0e-3, 1.0e-2)]

    num_samples = 100
    batch_size = num_samples
    input_size = Nx + 1 + 1
    nn_model = FeatureNN(input_size=input_size, param_bounds=param_bounds).to(device)
    physics_layer = PM.PhysicsLayer(dx=dx, nx=Nx, nt=100).to(device)

    for param in physics_layer.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(nn_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    true_alpha = 1.50e-3
    true_velocity = 0.01
    true_r0 = 7.0e-2
    true_r1 = 7.0e-3
    noise_std = 0.003

    training_data = prepare_training_data(physics_layer, true_alpha, true_velocity,
                                          true_r0, true_r1, num_samples, noise_std)

    epochs = 200
    loss_history = []
    relative_loss_history = []
    alpha_history = []
    velocity_history = []
    r0_history = []
    r1_history = []

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        training_data = training_data[perm]

        epoch_loss = 0.0
        relative_loss = 0.0
        alpha_vals = []
        velocity_vals = []
        r0_vals = []
        r1_vals = []

        for i in range(0, num_samples, batch_size):
            batch = training_data[i:i+batch_size].to(device)
            u_batch = batch[:, :-1]
            t_batch = batch[:, -1]

            optimizer.zero_grad()
            pred_params    = nn_model(batch)
            alpha_batch    = pred_params[:, 0]
            velocity_batch = pred_params[:, 1]
            r0_batch       = pred_params[:, 2]
            r1_batch       = pred_params[:, 3]

            output_batch = physics_layer.forward_solver(alpha_batch.mean(), velocity_batch.mean(), 
                                                        r0_batch.mean(), r1_batch.mean(), t_batch)

            loss = loss_fn(output_batch, u_batch)
            loss.backward()
            optimizer.step()

            batch_size_actual = batch.shape[0]
            epoch_loss += loss.item() * batch_size_actual

            with torch.no_grad():
                denom = torch.mean(u_batch.detach() ** 2).item()
            relative_loss += (loss.item() / denom) * batch_size_actual

            alpha_vals.extend(alpha_batch.tolist())
            velocity_vals.extend(velocity_batch.tolist())
            r0_vals.extend(r0_batch.tolist())
            r1_vals.extend(r1_batch.tolist())

        loss_history.append(epoch_loss / num_samples)
        relative_loss_history.append(relative_loss / num_samples)
        alpha_history.append(np.mean(alpha_vals))
        velocity_history.append(np.mean(velocity_vals))
        r0_history.append(np.mean(r0_vals))
        r1_history.append(np.mean(r1_vals))

        print(f"Epoch {epoch+1}/{epochs}, MSE = {loss_history[-1]:.6f}, Relative MSE = {relative_loss_history[-1]:.6f}")

    pred_params = nn_model(training_data).detach()
    pred_alpha_mean = pred_params[:, 0].mean().item()
    pred_velocity_mean = pred_params[:, 1].mean().item()
    pred_r0_mean = pred_params[:, 2].mean().item()
    pred_r1_mean = pred_params[:, 3].mean().item()

    print("\n--- Recovered Parameters ---")
    print(f"True alpha = {true_alpha:.3f}, Predicted alpha (mean) = {pred_alpha_mean:.4f}")
    print(f"True velocity  = {true_velocity:.3f}, Predicted velocity  (mean) = {pred_velocity_mean:.4f}")
    print(f"True r0  = {true_r0:.3f}, Predicted r0  (mean) = {pred_r0_mean:.4f}")
    print(f"True r1  = {true_r1:.3f}, Predicted r1  (mean) = {pred_r1_mean:.4f}")

    plt.figure(figsize=(14, 8))
    plt.subplot(2, 3, 1)
    plt.plot(loss_history, label='MSE')
    plt.plot(relative_loss_history, label='Relative MSE')
    plt.title("Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(alpha_history, label='Predicted α')
    plt.axhline(true_alpha, color='r', linestyle='--', label='True α')
    plt.title("Alpha Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Mean α")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(velocity_history, label='Predicted β')
    plt.axhline(true_velocity, color='r', linestyle='--', label='True β')
    plt.title("Beta Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Mean β")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(r0_history, label='Predicted r0')
    plt.axhline(true_r0, color='r', linestyle='--', label='True r0')
    plt.title("R0 Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Mean r0")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(r1_history, label='Predicted r1')
    plt.axhline(true_r1, color='r', linestyle='--', label='True r1')
    plt.title("R1 Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Mean r1")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

main()

