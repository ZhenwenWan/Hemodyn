import sys
import numpy as np
import scipy.sparse as sp
from CoDoSol import CoDoSol
import plots

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
        dt = self.dt
        dx = self.dx
        nx = self.nx
        try:
            A, Q, P = self.unpack(U)
            α     = self.params['alpha']
            ρ     = self.params['rho']
            β     = self.params['beta']
            KR    = self.params['K_R']
            Miu   = self.params['Miu']
            R_out = self.params['R_out']
            P_ref = self.params['P_ref']
            A0    = self.params['A0']
            Q0    = self.params['Q0']
            P0    = self.params['P0']

            R = np.zeros((nx, 3))
            # Dimensionless boundary conditions
            Aval_in = (np.sqrt(A0) + (self.P_in(t_curr) - P0) / β)**2
            R[0]  = [ A[0] - Aval_in,
                      Q[0] - self.Q_in(t_curr),
                      P[0] - self.P_in(t_curr)]
            R[-1] = [(A[-1] - A_prev[-1]) / dt + (Q[-1] - Q[-2]) / dx,
                      P[-1] - (P_ref + R_out * Q[-1]),
                      P[-1] - P0 - β * (np.sqrt(A[-1]) - np.sqrt(A0))]

            for i in range(1, nx-1):
                At   = (A[i] - A_prev[i]) / dt
                Ax   = (Q[i+1] - Q[i-1]) / (2 * dx)
                Qt   = (Q[i] - Q_prev[i]) / dt
                QQ   = α * Q[i+1]**2 / A[i+1]
                Qm   = Miu * (Q[i+1] - 2 * Q[i] + Q[i-1]) / dx / dx
                Mx   = (QQ - α * Q[i-1]**2 / A[i-1]) / (2 * dx)
                Pr   = A[i] * (P[i+1] - P[i-1]) / (2 * dx * ρ)
                R[i] = [At + Ax,
                        Qt + Mx + Pr + KR * Q[i] / A[i] - Qm,
                        P[i] - P0 - β * (np.sqrt(A[i]) - np.sqrt(A0))]

            for i in range(nx):
                R[i] = [R[i,0]/A0, R[i,1]/Q0, R[i,2]/P0]

            if np.any(np.isnan(R)) or np.any(np.isinf(R)):
                print(f"NaN or Inf in residual at t={t_curr:.3f}, R min/max: {np.min(R):.2e}/{np.max(R):.2e}")
            return R.T.ravel()
        except Exception as e:
            print(f"Error in sequential residual at t={t_curr:.3f}: {e}")
            raise

    def residual1(self, U, A_prev, Q_prev, P_prev, t_curr):
        dt = self.dt
        dx = self.dx
        nx = self.nx
        try:
            A, Q, P = self.unpack(U)
            α     = self.params['alpha']
            ρ     = self.params['rho']
            β     = self.params['beta']
            KR    = self.params['K_R']
            Miu   = self.params['Miu']
            R_out = self.params['R_out']
            P_ref = self.params['P_ref']
            A0    = self.params['A0']
            Q0    = self.params['Q0']
            P0    = self.params['P0']

            R = np.zeros((nx, 3))
            # Dimensionless boundary conditions
            Aval_in = (np.sqrt(A0) + (self.P_in(t_curr) - P0) / β)**2
            R[0]  = [ A[0] - Aval_in,
                      Q[0] - self.Q_in(t_curr),
                      P[0] - self.P_in(t_curr)]
            R[-1] = [(A[-1] - A_prev[-1]) / dt + (Q[-1] - Q[-2]) / dx,
                      P[-1] - (P_ref + R_out * Q[-1]),
                      P[-1] - P0 - β * (np.sqrt(A[-1]) - np.sqrt(A0))]

            for i in range(1, nx-1):
                At   = (A[i] - A_prev[i]) / dt
                Ax   = (Q[i+1] - Q[i-1]) / (2 * dx)
                Qt   = (Q[i] - Q_prev[i]) / dt
                QQ   = α * Q[i+1]**2 / A[i+1]
                Qm   = Miu * (Q[i+1] - 2 * Q[i] + Q[i-1]) / dx / dx
                Mx   = (QQ - α * Q[i-1]**2 / A[i-1]) / (2 * dx)
                Pr   = A[i] * (P[i+1] - P[i-1]) / (2 * dx * ρ)
                R[i] = [At + Ax,
                        Qt + Mx + Pr + KR * Q[i] / A[i] - Qm,
                        P[i] - P0 - β * (np.sqrt(A[i]) - np.sqrt(A0))]
                #print(f"i {i}, R[i][0] {R[i][0]:.3e}, R[i][1] {R[i][1]:.3e}, R[i][2] {R[i][2]:.3e}")
                #print(f"i {i}, A[i] {A[i]:.3e},  At {At:.3e},  Ax {Ax:.3e},  Q[i] {Q[i]:.3e},  P[i] {P[i]:.3e} ")


            for i in range(nx):
                R[i] = [R[i,0]/A0, R[i,1]/Q0, R[i,2]/P0]

            if np.any(np.isnan(R)) or np.any(np.isinf(R)):
                print(f"NaN or Inf in residual at t={t_curr:.3f}, R min/max: {np.min(R):.2e}/{np.max(R):.2e}")
            return R.T.ravel()
        except Exception as e:
            print(f"Error in sequential residual at t={t_curr:.3f}: {e}")
            raise

    def jacobian_crs(self, U, t_curr):

        def idx(i, k):
            return self.nx*k + i

        dt = self.dt
        dx = self.dx
        nx = self.nx
        try:
            A, Q, P = self.unpack(U)
            α     = self.params['alpha']
            ρ     = self.params['rho']
            β     = self.params['beta']
            KR    = self.params['K_R']
            Miu   = self.params['Miu']
            R_out = self.params['R_out']
            P_ref = self.params['P_ref']
            A0    = self.params['A0']
            Q0    = self.params['Q0']
            P0    = self.params['P0']
            rows, cols, data = [], [], []

            def add(r, c, v):
                rows.append(r); cols.append(c); data.append(v)

            add(idx(0, 0), idx(0, 0), 1.0/A0)
            add(idx(0, 1), idx(0, 1), 1.0/Q0)
            add(idx(0, 2), idx(0, 2), 1.0/P0)
            i = nx - 1
            add(idx(i, 0), idx(i, 0),      1.0/dt/A0) #(A[-1] - A_prev[-1]) / dt
            add(idx(i, 0), idx(i, 1),      1.0/dx/A0) #(Q[-1] - Q[-2]) / dx
            add(idx(i, 0), idx(i - 1, 1), -1.0/dx/A0) #(Q[-1] - Q[-2]) / dx
            add(idx(i, 1), idx(i, 1), -R_out/Q0) #-(P_ref + R_out * Q[-1])
            add(idx(i, 1), idx(i, 2),             1/Q0) #P[-1]
            add(idx(i, 2), idx(i, 0), -0.5*β/np.sqrt(A[-1])/P0) #-β * (np.sqrt(A[-1])
            add(idx(i, 2), idx(i, 2),                            1/P0) #P[-1]

            for i in range(1, nx-1):
                aji = A[i]
                qji = Q[i]
                pjp = P[i + 1]
                pjm = P[i - 1]
                ajp = A[i + 1]
                ajm = A[i - 1]
                qjp = Q[i + 1]
                qjm = Q[i - 1]

                r0 = idx(i, 0)
                add(r0, idx(i, 0),      1.0 / dt /A0)
                add(r0, idx(i + 1, 1),  1.0 / (2 * dx) /A0)
                add(r0, idx(i - 1, 1), -1.0 / (2 * dx) /A0)

                r1 = idx(i, 1)
                add(r1, idx(i, 1), 1.0 / dt /Q0 + 2 * Miu / dx / dx /Q0)
                add(r1, idx(i + 1, 1),  α * 2 * qjp    / ajp    / (2 * dx) /Q0 - Miu / dx / dx /Q0) 
                add(r1, idx(i + 1, 0), -α *     qjp**2 / ajp**2 / (2 * dx) /Q0)
                add(r1, idx(i - 1, 1), -α * 2 * qjm    / ajm    / (2 * dx) /Q0 - Miu / dx / dx /Q0)
                add(r1, idx(i - 1, 0),  α *     qjm**2 / ajm**2 / (2 * dx) /Q0)
                add(r1, idx(i + 1, 2),  aji / (2 * dx * ρ) /Q0)
                add(r1, idx(i - 1, 2), -aji / (2 * dx * ρ) /Q0)
                dpdx = (pjp - pjm) / (2 * dx)
                add(r1, idx(i, 0), dpdx / ρ /Q0)
                add(r1, idx(i, 0), -KR * qji / aji**2 /Q0)
                add(r1, idx(i, 1),  KR / aji /Q0)

                r2 = idx(i, 2)
                add(r2, idx(i, 0), -β / (2 * np.sqrt(aji))/P0)
                add(r2, idx(i, 2), 1.0/P0)

            J = sp.csr_matrix((data, (rows, cols)), shape=(3 * nx, 3 * nx))
            if np.any(np.isnan(J.data)) or np.any(np.isinf(J.data)):
                print(f"NaN or Inf in Jacobian at t={t_curr:.3f}, J min/max: {np.min(J.data):.2e}/{np.max(J.data):.2e}")
            return J
        except Exception as e:
            print(f"Error in sequential jacobian_crs at t={t_curr:.3f}: {e}")
            raise

    def explicit_predictor(self, U_prev, t_prev):
        dt = self.dt
        dx = self.dx
        nx = self.nx
        α            = self.params['alpha']
        ρ     = self.params['rho']
        β     = self.params['beta']
        KR    = self.params['K_R']
        Miu   = self.params['Miu']
        R_out = self.params['R_out']
        P_ref = self.params['P_ref']
        A0           = self.params['A0']
        Q0           = self.params['Q0']
        P0           = self.params['P0']
        A_prev, Q_prev, P_prev = self.unpack(U_prev)
        A_next = np.zeros_like(A_prev)
        Q_next = np.zeros_like(Q_prev)
        P_next = np.zeros_like(P_prev)
    
        # Boundary conditions
        Q_next[0] = self.Q_in(t_prev)
        P_next[0] = self.P_in(t_prev)
        A_next[0] = (np.sqrt(A0) + (self.P_in(t_prev) - P0) / β)**2

        for i in range(1, nx - 1):
            At = -(Q_prev[i+1] - Q_prev[i-1]) / (2 * dx)
            QQ_p = α * Q_prev[i+1]**2 / A_prev[i+1]
            QQ_m = α * Q_prev[i-1]**2 / A_prev[i-1]
            Mx = (QQ_p - QQ_m) / (2 * dx)
            Pr = A_prev[i] * (P_prev[i+1] - P_prev[i-1]) / (2 * dx * ρ)
            Qt = -(Mx + Pr + KR * Q_prev[i] / A_prev[i]) + Miu * (Q_prev[i+1] - 2 * Q_prev[i] + Q_prev[i-1]) / dx / dx
            A_next[i] = A_prev[i] + dt * At
            Q_next[i] = Q_prev[i] + dt * Qt
            P_next[i] = P0 + β * (np.sqrt(A_next[i]) - np.sqrt(A0))

        A_next[-1] = A_prev[-1] - (Q_prev[-1] - Q_prev[-2]) * dt / dx 
        P_next[-1] = P0 + β * (np.sqrt(A_next[-1]) - np.sqrt(A0))
        Q_next[-1] = (P_next[-1] - P_ref) / R_out
    
        U_pred = np.concatenate([A_next, Q_next, P_next])
        return U_pred

    def solve_step(self, U_prev, t_curr, tol=1e-6, maxit=300):
        try:
            A_prev, Q_prev, P_prev = self.unpack(U_prev)
            def fun(U):
                return self.residual(U, A_prev, Q_prev, P_prev, t_curr)
            def fun1(U):
                return self.residual1(U, A_prev, Q_prev, P_prev, t_curr)
    
            def jac(U):
                return self.jacobian_crs(U, t_curr).toarray()
    
            if self.bounds:
                lb, ub = self.bounds
            else:
                lb = np.full_like(U_prev, -np.inf)
                ub = np.full_like(U_prev, np.inf)
    
            tol = [tol, 0]
            parms = [maxit, 3000, 1, -1, 0, 2]

            U_Euler = self.explicit_predictor(U_prev, t_curr)

            A, Q, P = self.unpack(U_Euler)
            x = np.linspace(0, 1, self.nx)
            #plots.display1step(1, x, A, Q, P)

            U_Euler = np.clip(U_Euler, lb + 1e-6, ub - 1e-6)
            result  = CoDoSol(U_Euler, fun, lb, ub, tol, parms, jac)
            sol, ierr, output, _, _, _ = result
            print(f"t={t_curr:.3f}, Ierr: {ierr}, Output: {output}")

            A, Q, P = self.unpack(sol)
            print(f"Q = {Q[0]:.6f}, {Q[1]:.6f}, {Q[2]:.6f}, {Q[3]:.6f}")
            U = np.concatenate([A, Q, P])
            #res = fun1( U )
            #print(f"np.max(Q) = {np.max(Q):.6f}")
            #print(f"residual  = {np.max(res):.6e}")
            plots.display1step(2, x, A, Q, P)


            return sol, ierr
        except Exception as e:
            print(f"Error in solve_step at t={t_curr:.3f}: {e}")
            return None

    def solve(self, U0, x_, nt_, t_, tol=1e-6, maxit=3000):
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
                U_next, status = self.solve_step(U_guess, t_curr, tol, maxit)
                A, Q, P = self.unpack(U_next)
                A_seq[j], Q_seq[j], P_seq[j] = A, Q, P
                U_prev = U_next
                #plots.display1step(j, x_, A, Q, P)
                print(f"Completed step {j}/{nt_-1}, t={t_curr:.3f}")
            return A_seq, Q_seq, P_seq
        except Exception as e:
            print(f"Error in sequential solve: {e}")
            return A_seq[:j], Q_seq[:j], P_seq[:j]

