import sys
import numpy as np
import scipy.sparse as sp
from CoDoSol import CoDoSol

class SequentialHemodynamics1D:
    def __init__(self, nx_, dx_, dt_, params_, Q_in_, P_in_, P_out_, Q_out_, bounds_=None):
        try:
            self.nx = nx_
            self.dx, self.dt = dx_, dt_
            self.params = params_
            self.Q_in, self.P_in, self.P_out, self.Q_out = Q_in_, P_in_, P_out_, Q_out_
            self.bounds = bounds_
            self.size = 3 * nx_
            print("SequentialHemodynamics1D initialized")
        except Exception as e:
            print("Error in SequentialHemodynamics1D __init__:", e)
            raise

    def unpack(self, U):
        try:
            A = U[:self.nx]  # Dimensionless A
            Q = U[self.nx:2*self.nx]  # Dimensionless Q
            P = U[2*self.nx:3*self.nx]  # Dimensionless P
            return A, Q, P
        except Exception as e:
            print("Error in unpack:", e)
            raise

    def residual(self, U, A_prev, Q_prev, P_prev, t_curr):
        try:
            A, Q, P = self.unpack(U)
            α = self.params['alpha']
            ρ_scaled = self.params['rho_scaled']
            β_scaled = self.params['beta_scaled']
            KR_scaled = self.params['K_R_scaled']
            R = np.zeros((self.nx, 3))
            # Dimensionless boundary conditions
            Aval_in = ((self.P_in(t_curr) - (1 - β_scaled)) / β_scaled)**2
            Aval_out = ((self.P_out(t_curr) - (1 - β_scaled)) / β_scaled)**2
            if Aval_in <= 0 or Aval_out <= 0:
                print(f"Warning: Invalid boundary conditions at t={t_curr:.3f}, Aval_in={Aval_in:.2e}, Aval_out={Aval_out:.2e}")
            R[0] = [A[0] - Aval_in,
                    Q[0] - self.Q_in(t_curr),
                    P[0] - self.P_in(t_curr)]
            R[-1] = [A[-1] - Aval_out,
                     Q[-1] - self.Q_out(t_curr),
                     P[-1] - self.P_out(t_curr)]

            for i in range(1, self.nx-1):
                At = (A[i] - A_prev[i]) / self.dt
                Ax = (Q[i+1] - Q[i-1]) / (2 * self.dx)
                Qt = (Q[i] - Q_prev[i]) / self.dt
                QQ = α * Q[i+1]**2 / A[i+1]
                Mx = (QQ - α * Q[i-1]**2 / A[i-1]) / (2 * self.dx)
                Pr = A[i] * (P[i+1] - P[i-1]) / (2 * self.dx * ρ_scaled)
                C = 1.0 + β_scaled * (np.sqrt(A[i]) - 1.0)
                R[i] = [At + Ax,
                        Qt + Mx + Pr + KR_scaled * Q[i] / A[i],
                        P[i] - C]

            if np.any(np.isnan(R)) or np.any(np.isinf(R)):
                print(f"NaN or Inf in residual at t={t_curr:.3f}, R min/max: {np.min(R):.2e}/{np.max(R):.2e}")
            return R.T.ravel()
        except Exception as e:
            print(f"Error in sequential residual at t={t_curr:.3f}: {e}")
            raise

    def jacobian_crs(self, U, t_curr):

        def idx(i, k):
            return self.nx*k + i

        try:
            A, Q, P = self.unpack(U)
            α = self.params['alpha']
            ρ_scaled = self.params['rho_scaled']
            β_scaled = self.params['beta_scaled']
            KR_scaled = self.params['K_R_scaled']
            rows, cols, data = [], [], []

            def add(r, c, v):
                rows.append(r); cols.append(c); data.append(v)

            add(idx(0, 0), idx(0, 0), 1.0)
            add(idx(0, 1), idx(0, 1), 1.0)
            add(idx(0, 2), idx(0, 2), 1.0)
            add(idx(self.nx-1, 0), idx(self.nx-1, 0), 1.0)
            add(idx(self.nx-1, 1), idx(self.nx-1, 1), 1.0)
            add(idx(self.nx-1, 2), idx(self.nx-1, 2), 1.0)

            for i in range(1, self.nx-1):
                aji = A[i]
                qji = Q[i]
                pjp = P[i + 1]
                pjm = P[i - 1]
                ajp = A[i + 1]
                ajm = A[i - 1]
                qjp = Q[i + 1]
                qjm = Q[i - 1]

                r0 = idx(i, 0)
                add(r0, idx(i, 0), 1.0 / self.dt)
                add(r0, idx(i + 1, 1), 1.0 / (2 * self.dx))
                add(r0, idx(i - 1, 1), -1.0 / (2 * self.dx))

                r1 = idx(i, 1)
                add(r1, idx(i, 1), 1.0 / self.dt)
                add(r1, idx(i + 1, 1), α * 2 * qjp / ajp / (2 * self.dx))
                add(r1, idx(i + 1, 0), -α * qjp**2 / ajp**2 / (2 * self.dx))
                add(r1, idx(i - 1, 1), -α * 2 * qjm / ajm / (2 * self.dx))
                add(r1, idx(i - 1, 0), α * qjm**2 / ajm**2 / (2 * self.dx))
                add(r1, idx(i + 1, 2), aji / (2 * self.dx * ρ_scaled))
                add(r1, idx(i - 1, 2), -aji / (2 * self.dx * ρ_scaled))
                dpdx = (pjp - pjm) / (2 * self.dx)
                add(r1, idx(i, 0), dpdx / ρ_scaled)
                add(r1, idx(i, 1), KR_scaled / aji)
                add(r1, idx(i, 0), -KR_scaled * qji / aji**2)

                r2 = idx(i, 2)
                add(r2, idx(i, 2), 1.0)
                add(r2, idx(i, 0), -β_scaled / (2 * np.sqrt(aji)))

            J = sp.csr_matrix((data, (rows, cols)), shape=(3 * self.nx, 3 * self.nx))
            if np.any(np.isnan(J.data)) or np.any(np.isinf(J.data)):
                print(f"NaN or Inf in Jacobian at t={t_curr:.3f}, J min/max: {np.min(J.data):.2e}/{np.max(J.data):.2e}")
            return J
        except Exception as e:
            print(f"Error in sequential jacobian_crs at t={t_curr:.3f}: {e}")
            raise

    def explicit_predictor(self, U_prev, t_prev):
        A_prev, Q_prev, P_prev = self.unpack(U_prev)
        A_next = np.zeros_like(A_prev)
        Q_next = np.zeros_like(Q_prev)
        P_next = np.zeros_like(P_prev)
    
        α = self.params['alpha']
        ρ_scaled = self.params['rho_scaled']
        β_scaled = self.params['beta_scaled']
        KR_scaled = self.params['K_R_scaled']
    
        # Boundary conditions
        Aval_in = max(((self.P_in(t_prev) - (1 - β_scaled)) / β_scaled)**2, 1.0)
        Aval_out = max(((self.P_out(t_prev) - (1 - β_scaled)) / β_scaled)**2, 1.0)
        Q_next[0] = self.Q_in(t_prev)
        Q_next[-1] = self.Q_out(t_prev)
        A_next[0] = Aval_in
        A_next[-1] = Aval_out
        P_next[0] = self.P_in(t_prev)
        P_next[-1] = self.P_out(t_prev)
        for i in range(1, self.nx - 1):
            At = -(Q_prev[i+1] - Q_prev[i-1]) / (2 * self.dx)
            QQ_p = α * Q_prev[i+1]**2 / A_prev[i+1]
            QQ_m = α * Q_prev[i-1]**2 / A_prev[i-1]
            Mx = (QQ_p - QQ_m) / (2 * self.dx)
            Pr = A_prev[i] * (P_prev[i+1] - P_prev[i-1]) / (2 * self.dx * ρ_scaled)
            Qt = -(Mx + Pr + KR_scaled * Q_prev[i] / A_prev[i])
            A_next[i] = A_prev[i] + self.dt * At
            Q_next[i] = Q_prev[i] + self.dt * Qt
            P_next[i] = 1.0 + β_scaled * (np.sqrt(A_next[i]) - 1.0)
    
        U_pred = np.concatenate([A_next, Q_next, P_next])
        return U_pred

    def solve_step(self, U_prev, t_curr, tol=1e-6, maxit=300):
        try:
            A_prev, Q_prev, P_prev = self.unpack(U_prev)
            def fun(U):
                return self.residual(U, A_prev, Q_prev, P_prev, t_curr)
    
            def jac(U):
                return self.jacobian_crs(U, t_curr).toarray()
    
            if self.bounds:
                lb, ub = self.bounds
            else:
                lb = np.full_like(U_prev, -np.inf)
                ub = np.full_like(U_prev, np.inf)
    
            tol = [tol, 0]
            parms = [maxit, 1000, 1, -1, 0, 2]

            U_Euler = self.explicit_predictor(U_prev, t_curr)
            U_Euler = np.clip(U_Euler, lb + 1e-6, ub - 1e-6)
            result  = CoDoSol(U_Euler, fun, lb, ub, tol, parms, jac)

            if len(result) == 3:
                print(f"Solver failed at t={t_curr:.3f} due to infeasible initial guess")
                return None
            sol, ierr, output, _, _, _ = result
            print(f"t={t_curr:.3f}, Ierr: {ierr}, Output: {output}")
            if ierr > 0:
                J_dense = jac(U_Euler)
                cond_number = np.linalg.cond(J_dense)
                print(f"Jacobian condition number at t={t_curr:.3f}: {cond_number:.2e}")

                U_Euler = 0.5 * ( U_prev + self.explicit_predictor(U_prev, t_curr) )
                U_Euler = np.clip(U_Euler, lb + 1e-6, ub - 1e-6)
                result  = CoDoSol(U_Euler, fun, lb, ub, tol, parms, jac)

                if len(result) == 3:
                    print(f"Solver failed at t={t_curr:.3f} due to infeasible initial guess")
                    return None
                sol, ierr, output, _, _, _ = result
                print(f"t={t_curr:.3f}, Ierr: {ierr}, Output: {output}")

            return sol, ierr
        except Exception as e:
            print(f"Error in solve_step at t={t_curr:.3f}: {e}")
            return None

    def solve(self, U0, x_, nt_, t_, tol=1e-6, maxit=300):
        try:
            A_seq = np.zeros((nt_, self.nx))
            Q_seq = np.zeros((nt_, self.nx))
            P_seq = np.zeros((nt_, self.nx))
            U_prev = U0.copy()  # Ensure writable copy
            A_seq[0], Q_seq[0], P_seq[0] = self.unpack(U_prev)
            print("Starting sequential solver")
            for j in range(1, nt_):
                t_curr = t_[j]
                U_guess = U_prev.copy()  # Ensure writable copy
                A_guess, Q_guess, P_guess = self.unpack(U_guess)
                U_next, status = self.solve_step(U_guess, t_curr, tol, maxit)
                A, Q, P = self.unpack(U_next)
                A_seq[j], Q_seq[j], P_seq[j] = A, Q, P
                U_prev = U_next
                print(f"Completed step {j}/{nt_-1}, t={t_curr:.3f}")

            return A_seq, Q_seq, P_seq
        except Exception as e:
            print(f"Error in sequential solve: {e}")
            return A_seq[:j], Q_seq[:j], P_seq[:j]

