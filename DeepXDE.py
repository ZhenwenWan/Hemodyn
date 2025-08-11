import deepxde as dde
import numpy as np

# 1) Space–time domain
L, T = 0.2, 1.0                      # artery length [m], time horizon [s]
geom = dde.geometry.Interval(0.0, L)
timed = dde.geometry.TimeDomain(0.0, T)
geomtime = dde.geometry.GeometryXTime(geom, timed)

# 2) Unknowns: y = [A(x,t), Q(x,t)]
rho = 1060.0
A0  = 3.14e-5                         # reference area
beta = dde.Variable(1.0e5)            # trainable wall stiffness
Cf   = dde.Variable(1.0e2)            # trainable friction coeff

def P_of_A(A):
    # Example elastic tube: P = beta*(sqrt(A) - sqrt(A0))/A0
    return beta * (dde.backend.sqrt(A) - np.sqrt(A0)) / A0

def dPdA(A):
    return beta * (0.5 / (A0 * dde.backend.sqrt(A) + 1e-12))

def pde(X, y):
    x, t = X[:, 0:1], X[:, 1:2]
    A, Q = y[:, 0:1], y[:, 1:2]

    A_t = dde.grad.jacobian(y, X, i=0, j=1)
    Q_t = dde.grad.jacobian(y, X, i=1, j=1)
    A_x = dde.grad.jacobian(y, X, i=0, j=0)
    Q_x = dde.grad.jacobian(y, X, i=1, j=0)

    P_x = dPdA(A) * A_x

    r1 = A_t + Q_x                                           # continuity
    r2 = Q_t + dde.grad.jacobian(Q*Q/A, X, i=0, j=0) + A/rho * P_x + Cf*Q/A
    return [r1, r2]

# 3) ICs (set your own)
ic_A = dde.icbc.IC(geomtime, lambda X: A0*np.ones_like(X[:, :1]), lambda _, on_initial: on_initial, component=0)
ic_Q = dde.icbc.IC(geomtime, lambda X: 0.0*np.ones_like(X[:, :1]),  lambda _, on_initial: on_initial, component=1)

# 4) BCs (examples)
left = lambda X, on_b: on_b and dde.utils.isclose(X[0], 0.0)
right = lambda X, on_b: on_b and dde.utils.isclose(X[0], L)

# Inlet flow waveform Q_in(t) if known:
Q_in = lambda X: 1e-5*np.sin(2*np.pi*X[:,1:2]/T).clip(min=0)
bc_Q_in = dde.icbc.DirichletBC(geomtime, Q_in, left, component=1)

# Outlet pressure via Windkessel (illustration) using OperatorBC:
def windkessel(inputs, outputs, X):
    # P + RC dP/dt = R Q   at x = L
    A, Q = outputs[:,0:1], outputs[:,1:2]
    P = P_of_A(A)
    P_t = dPdA(A) * dde.grad.jacobian(outputs, inputs, i=0, j=1)  # dA/dt * dP/dA
    R, C = 1.0e7, 1.0e-9
    return P + R*C*P_t - R*Q
bc_out = dde.icbc.OperatorBC(geomtime, windkessel, right)

# 5) Data + network
data = dde.data.TimePDE(geomtime, pde, [ic_A, ic_Q, bc_Q_in, bc_out],
                        num_domain=5000, num_boundary=200, num_initial=200)
net = dde.nn.FNN([2] + [64]*4 + [2], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3,
              external_trainable_variables=[beta, Cf])
model.train(iterations=30000)
model.compile("L-BFGS", external_trainable_variables=[beta, Cf])
model.train()

