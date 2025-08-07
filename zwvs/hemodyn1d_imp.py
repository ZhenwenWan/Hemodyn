import numpy as np
import scipy.sparse as sp
from CoDoSol import CoDoSol

class ImplicitHemodynamics1D:
    def __init__(self, nx, nt, dx, dt, params, Q_in, P_in, P_out, Q_out, bounds=None):
        self.nx, self.nt = nx, nt
        self.dx, self.dt = dx, dt
        self.params = params
        self.Q_in, self.P_in, self.P_out, self.Q_out = Q_in, P_in, P_out, Q_out
        self.bounds = bounds
        self.N = nx * nt
        self.size = 3 * self.N
        print("ImplicitHemodynamics1D initialized")

    def idx(self, j, i, k):
        return k * self.N + j * self.nx + i

    def unpack(self, U):
        A = U[:self.N].reshape(self.nt, self.nx)
        Q = U[self.N:2*self.N].reshape(self.nt, self.nx)
        P = U[2*self.N:3*self.N].reshape(self.nt, self.nx)
        return A, Q, P

    def residual(self, U):
        A, Q, P = self.unpack(U)
        α, ρ, β = self.params['alpha'], self.params['rho'], self.params['beta']
        P0, A0, Q0, KR = self.params['P0'], self.params['A0'], self.params['Q0'], self.params['K_R']

        R = np.zeros((self.nt, self.nx, 3))

        # Initial condition
        for i in range(self.nx):
            R[0, i] = [(A[0, i] - A0) / A0,
                       (Q[0, i] - self.Q_in(0)) / Q0,
                       (P[0, i] - P0) / P0]

        for j in range(1, self.nt - 1):
            for i in range(1, self.nx - 1):
                At = (A[j+1, i] - A[j-1, i]) / (2 * self.dt)
                Ax = (Q[j, i+1] - Q[j, i-1]) / (2 * self.dx)
                Qt = (Q[j+1, i] - Q[j-1, i]) / (2 * self.dt)
                QQp = α * Q[j, i+1]**2 / A[j, i+1]
                QQm = α * Q[j, i-1]**2 / A[j, i-1]
                Mx = (QQp - QQm) / (2 * self.dx)
                Pr = A[j, i] * (P[j, i+1] - P[j, i-1]) / (2 * self.dx * ρ)
                C = P0 + β * (np.sqrt(A[j, i]) - np.sqrt(A0))
                R[j, i] = [At + Ax,
                           Qt + Mx + Pr + KR * Q[j, i] / A[j, i],
                           P[j, i] - C]

        # Terminal periodicity
        for i in range(self.nx):
            R[-1, i] = [(A[-1, i] - A[0, i]) / A0,
                        (Q[-1, i] - Q[0, i]) / Q0,
                        (P[-1, i] - P[0, i]) / P0]

        # Boundary conditions
        penalty = 1e-2 * A0
        for j in range(1, self.nt - 1):
            t = j * self.dt
            Aval_in = ((self.P_in(t) - P0) / β + np.sqrt(A0))**2
            Aval_out = ((self.P_out(t) - P0) / β + np.sqrt(A0))**2
            R[j, 0] = [(A[j, 0] - Aval_in) * penalty / A0,
                       (Q[j, 0] - self.Q_in(t)) * penalty / Q0,
                       (P[j, 0] - self.P_in(t)) * penalty / P0]
            R[j, -1] = [(A[j, -1] - Aval_out) * penalty / A0,
                        (Q[j, -1] - self.Q_out(t)) * penalty / Q0,
                        (P[j, -1] - self.P_out(t)) * penalty / P0]

        return R.ravel()

    def analytical_jacobian_crs(U, unpack, params, nx, nt, dx, dt, idx_fn):
        A, Q, P = unpack(U)
        α, ρ, β = params['alpha'], params['rho'], params['beta']
        P0, A0, Q0, KR = params['P0'], params['A0'], params['Q0'], params['K_R']
        N = nx * nt
    
        rows, cols, data = [], [], []
    
        def add(r, c, v):
            rows.append(r)
            cols.append(c)
            data.append(v)
    
        for j in range(1, nt - 1):
            for i in range(1, nx - 1):
                aji = A[j, i]
                qji = Q[j, i]
                ajp = A[j, i + 1]
                ajm = A[j, i - 1]
                qjp = Q[j, i + 1]
                qjm = Q[j, i - 1]
                pjp = P[j, i + 1]
                pjm = P[j, i - 1]
    
                # Continuity equation
                r0 = idx_fn(j, i, 0)
                add(r0, idx_fn(j + 1, i, 0),  1.0 / (2 * dt))
                add(r0, idx_fn(j - 1, i, 0), -1.0 / (2 * dt))
                add(r0, idx_fn(j, i + 1, 1),  1.0 / (2 * dx))
                add(r0, idx_fn(j, i - 1, 1), -1.0 / (2 * dx))
    
                # Momentum equation
                r1 = idx_fn(j, i, 1)
                add(r1, idx_fn(j + 1, i, 1),  1.0 / (2 * dt))
                add(r1, idx_fn(j - 1, i, 1), -1.0 / (2 * dt))
                add(r1, idx_fn(j, i + 1, 1),  α * 2 * qjp / ajp / (2 * dx))
                add(r1, idx_fn(j, i + 1, 0), -α * qjp**2 / ajp**2 / (2 * dx))
                add(r1, idx_fn(j, i - 1, 1), -α * 2 * qjm / ajm / (2 * dx))
                add(r1, idx_fn(j, i - 1, 0),  α * qjm**2 / ajm**2 / (2 * dx))
                add(r1, idx_fn(j, i + 1, 2),  aji / (2 * dx * ρ))
                add(r1, idx_fn(j, i - 1, 2), -aji / (2 * dx * ρ))
                dpdx = (pjp - pjm) / (2 * dx)
                add(r1, idx_fn(j, i, 0), dpdx / ρ)
                add(r1, idx_fn(j, i, 1), KR / aji)
                add(r1, idx_fn(j, i, 0), -KR * qji / aji**2)
    
                # State equation
                r2 = idx_fn(j, i, 2)
                add(r2, idx_fn(j, i, 2), 1.0)
                add(r2, idx_fn(j, i, 0), -β / (2 * np.sqrt(aji)))
    
        return sp.csr_matrix((data, (rows, cols)), shape=(3 * N, 3 * N))

    def jacobian_crs(self, U):
        return analytical_jacobian_crs(U, self.unpack, self.params, self.nx, self.nt, self.dx, self.dt, self.idx)

    def solve(self, U0, tol=1e-6, maxit=300):
        def fun(U): return self.residual(U)
        def jac(U): return self.jacobian_crs(U).toarray()

        lb = np.full_like(U0, -np.inf) if not self.bounds else self.bounds[0]
        ub = np.full_like(U0, np.inf) if not self.bounds else self.bounds[1]
        parms = [maxit, 2000, 1, -1, 1, 2]

        print("Solving full system with CoDoSol")
        result = CoDoSol(U0, fun, lb, ub, [tol, 0], parms, jac)
        if len(result) == 3:
            print("Solver failed")
            return None, None, None
        sol, ierr, *_ = result
        A, Q, P = self.unpack(sol)
        return A, Q, P

