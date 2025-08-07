import numpy as np
import matplotlib.pyplot as plt

def display_step_result(step, x, t, A, Q, P, A0, Q0, P0):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(x, A[step] * A0, label=f't={t[step]:.3f}')
    axs[0].set_title("A (Cross-sectional Area)")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("A (m²)")

    axs[1].plot(x, Q[step] * Q0, label=f't={t[step]:.3f}')
    axs[1].set_title("Q (Flow Rate)")
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("Q (m³/s)")

    axs[2].plot(x, P[step] * P0, label=f't={t[step]:.3f}')
    axs[2].set_title("P (Pressure)")
    axs[2].set_xlabel("x (m)")
    axs[2].set_ylabel("P (Pa)")

    for ax in axs:
        ax.legend()
        ax.grid(True)
    plt.suptitle(f"Solution at t={t[step]:.3f}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def display1step(step, x, A, Q, P):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].plot(x, A, label=f't={step}')
    axs[0].set_title("A (Cross-sectional Area)")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("A (m²)")

    axs[1].plot(x, Q, label=f't={step}')
    axs[1].set_title("Q (Flow Rate)")
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("Q (m³/s)")

    axs[2].plot(x, P, label=f't={step}')
    axs[2].set_title("P (Pressure)")
    axs[2].set_xlabel("x (m)")
    axs[2].set_ylabel("P (Pa)")

    for ax in axs:
        ax.legend()
        ax.grid(True)
    plt.suptitle(f"Solution at t={step}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def visualize_all_steps(x, t, A, Q, P, A0, Q0, P0):
    X, T = np.meshgrid(x, t)

    for name, data, scale, label in [
        ("A (Area)", A, A0, "A (m²)"),
        ("Q (Flow)", Q, Q0, "Q (m³/s)"),
        ("P (Pressure)", P, P0, "P (Pa)")
    ]:
        plt.figure(figsize=(10, 4))
        plt.contourf(X, T, data * scale, levels=50, cmap='viridis')
        plt.colorbar(label=label)
        plt.xlabel("x (m)")
        plt.ylabel("t (s)")
        plt.title(f"{name} over Space and Time")
        plt.tight_layout()
        plt.show()

