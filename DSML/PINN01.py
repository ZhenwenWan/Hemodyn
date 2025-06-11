import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------
# 1. Data Generation
# -------------------------------

def generate_cooling_curve(ambient_temp, noise_std=0.5):
    t = np.linspace(0, 60, 60)
    alpha = 0.05  # cooling rate constant
    T0 = 1000
    T = ambient_temp + (T0 - ambient_temp) * np.exp(-alpha * t)
    noise = np.random.normal(0, noise_std, size=T.shape)
    return T + noise

# Generate positive samples (same metal part)
ambient_temps_pos = np.linspace(0, 50, 51)  # 51 samples
X_pos = np.array([generate_cooling_curve(at) for at in ambient_temps_pos])
y_pos = np.ones(len(X_pos))

# Generate negative samples (general false curves)
ambient_temps_neg = np.random.uniform(0, 50, 500)
X_neg = []
for at in ambient_temps_neg:
    T0 = 1000
    T_end = at
    t = np.linspace(0, 60, 60)
    
    # Generate alternative cooling curve: different cooling rate, plus sinusoidal wiggles
    alpha_alt = np.random.uniform(0.025, 0.1)  # Different rate
    wiggle = 5 * np.sin(0.2 * t + np.random.uniform(0, 2*np.pi))
    T_random = T_end + (T0 - T_end) * np.exp(-alpha_alt * t) + wiggle
    
    # Enforce start and end
    T_random[0] = 1000
    T_random[-1] = T_end
    # Compare with classical positives
    is_positive = False
    for pos_curve in X_pos:
        if np.all(np.abs(T_random - pos_curve) < 1.0):
            is_positive = True
            break

    if not is_positive:
        X_neg.append(T_random)    

X_neg = np.array(X_neg)
y_neg = np.zeros(len(X_neg))

# Combine and split
X = np.vstack([X_pos, X_neg])
y = np.hstack([y_pos, y_neg])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# -------------------------------
# 2. Model Definitions
# -------------------------------

# Model A: General NN
class GeneralNN(nn.Module):
    def __init__(self):
        super(GeneralNN, self).__init__()
        self.fc1 = nn.Linear(60, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Model B: Physics-Aware NN
class PhysicsAwareNN(nn.Module):
    def __init__(self):
        super(PhysicsAwareNN, self).__init__()
        # Feature extractor: approximate ambient temp
        self.extractor = nn.Sequential(
            nn.Linear(60, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        ambient_est = self.extractor(x)
        output = self.classifier(ambient_est)
        return output

# -------------------------------
# 3. Training Loop with Accuracy
# -------------------------------

def train_model(model, X_train, y_train, X_val, y_val, epochs=50):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        with torch.no_grad():
            preds_train = (outputs > 0.5).float()
            acc_train = (preds_train == y_train).float().mean().item() * 100
            train_accuracies.append(acc_train)

            model.eval()
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
            preds_val = (val_outputs > 0.5).float()
            acc_val = (preds_val == y_val).float().mean().item() * 100
            val_accuracies.append(acc_val)

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], "
                  f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, "
                  f"Train Acc: {acc_train:.2f}%, Val Acc: {acc_val:.2f}%")
            
    return train_losses, val_losses, train_accuracies, val_accuracies

# -------------------------------
# 4. Run Experiments
# -------------------------------

model_A = GeneralNN()
model_B = PhysicsAwareNN()

print("Training General NN...")
train_losses_A, val_losses_A, train_accs_A, val_accs_A = train_model(model_A, X_train, y_train, X_val, y_val)

print("\nTraining Physics-Aware NN...")
train_losses_B, val_losses_B, train_accs_B, val_accs_B = train_model(model_B, X_train, y_train, X_val, y_val)

# -------------------------------
# 5. Plot Loss and Accuracy Curves
# -------------------------------

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses_A, label='General NN - Train Loss')
plt.plot(val_losses_A, label='General NN - Val Loss')
plt.plot(train_losses_B, label='Physics-Aware NN - Train Loss')
plt.plot(val_losses_B, label='Physics-Aware NN - Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1,2,2)
plt.plot(train_accs_A, label='General NN - Train Acc')
plt.plot(val_accs_A, label='General NN - Val Acc')
plt.plot(train_accs_B, label='Physics-Aware NN - Train Acc')
plt.plot(val_accs_B, label='Physics-Aware NN - Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.ylim([90, 100])
plt.show()

