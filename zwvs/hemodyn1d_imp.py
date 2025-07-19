class ImplicitHemodynamics1D:
    def __init__(self, nx_, nt_, dx_, dt_, params_, Q_in_, P_in_, P_out_, Q_out_, bounds_=None):
        try:
            self.nx, self.nt = nx_, nt_
            self.dx, self.dt = dx_, dt_
            self.params = params_
            self.Q_in, self.P_in, self.P_out, self.Q_out = Q_in_, P_in_, P_out_, Q_out_
            self.bounds = bounds_
            self.N = nx_ * nt_
            self.size = 3 * self.N
            print("ImplicitHemodynamics1D initialized")
        except Exception as e:
            print("Error in ImplicitHemodynamics1D __init__:", e)
            raise

    def idx(self, j, i, k):
        return k * self.N + j * self.nx + i

    def unpack(self, U):
        try:
            A = U[:self.N].reshape(self.nt, self.nx)  # SI units: m²
            Q = U[self.N:2*self.N].reshape(self.nt, self.nx)  # SI units: m³/s
            P = U[2*self.N:3*self.N].reshape(self.nt, self.nx)  # SI units: Pa
            return A, Q, P
        except Exception as e:
            print("Error in unpack:", e)
            raise

    def residual(self, U):
        try:
            A, Q, P = self.unpack(U)
            α, ρ, β = self.params['alpha'], self.params['rho'], self.params['beta']
            P0, A0, Q0, KR, Rpar = self.params['P0'], self.params['A0'], self.params['Q0'], self.params['K_R'], self.params['R']
            R = np.zeros((self.nt, self.nx, 3))

            for i in range(self.nx):
                R[0,i] = [(A[0,i]-A0)/A0,
                          (Q[0,i]-Q_in(0))/Q0,
                          (P[0,i]-P0)/P0]

            for j in range(1, self.nt-1):
                for i in range(1, self.nx-1):
                    At = (A[j+1,i]-A[j-1,i]) / (2*self.dt)
                    Ax = (Q[j,i+1]-Q[j,i-1]) / (2*self.dx)
                    Qt = (Q[j+1,i]-Q[j-1,i]) / (2*self.dt)
                    QQ = α * Q[j,i+1]**2 / A[j,i+1]
                    Mx = (QQ - α * Q[j,i-1]**2 / A[j,i-1]) / (2*self.dx)
                    Pr = (A[j,i]/ρ) * (P[j,i+1]-P[j,i-1]) / (2*self.dx)
                    C = P0 + β*(np.sqrt(A[j,i]) - np.sqrt(A0))
                    R[j,i] = [(At+Ax)/1e3,
                              (Qt + Mx + Pr + KR*Q[j,i]/A[j,i])/1e3,
                              (P[j,i]-C)/1e3]

            for i in range(self.nx):
                R[-1,i] = [(A[-1,i]-A[0,i])/A0,
                           (Q[-1,i]-Q[0,i])/Q0,
                           (P[-1,i]-P[0,i])/P0]

            for j in range(1, self.nt-1):
                Aval_in = ((P_in(j*self.dt)-P0)/β + np.sqrt(A0))**2
                Aval_out = ((P_out(j*self.dt)-P0)/β + np.sqrt(A0))**2
                penalty = 1e-2 * A0  # Scales to 1 for A, Q; 1e-7 for P
                R[j,0] = [(A[j,0]-Aval_in)*penalty/A0,
                          (Q[j,0]-Q_in(j*self.dt))*penalty/Q0,
                          (P[j,0]-P_in(j*self.dt))*penalty/P0]
                R[j,-1] = [(A[j,-1]-Aval_out)*penalty/A0,
                           (Q[j,-1]-Q_out(j*self.dt))*penalty/Q0,
                           (P[j,-1]-P_out(j*self.dt))*penalty/P0]

            if np.any(np.isnan(R)) or np.any(np.isinf(R)):
                print("NaN or Inf detected in residual, R min/max:", np.min(R), np.max(R))
            print("Residual norm (A):", np.linalg.norm(R[:,:,0].ravel()))
            print("Residual norm (Q):", np.linalg.norm(R[:,:,1].ravel()))
            print("Residual norm (P):", np.linalg.norm(R[:,:,2].ravel()))
            print("Residual norm (total):", np.linalg.norm(R.ravel()))
            return R.ravel()
        except Exception as e:
            print("Error in residual:", e)
            raise

    def jacobian_crs(self, U):
        try:
            A, Q, P = self.unpack(U)
            α, ρ, β = self.params['alpha'], self.params['rho'], self.params['beta']
            P0, A0, Q0, KR, Rpar = self.params['P0'], self.params['A0'], self.params['Q0'], self.params['K_R'], self.params['R']
            rows, cols, data = [], [], []

            def add(r, c, v):
                rows.append(r); cols.append(c); data.append(v)

            for i in range(self.nx):
                add(self.idx(0, i, 0), self.idx(0, i, 0), 1.0)
                add(self.idx(0, i, 1), self.idx(0, i, 1), 1.0)
                add(self.idx(0, i, 2), self.idx(0, i, 2), 1.0)

            for i in range(self.nx):
                for k in (0, 1, 2):
                    add(self.idx(self.nt-1, i, k), self.idx(self.nt-1, i, k), 1.0)
                    add(self.idx(self.nt-1, i, k), self.idx(0, i, k), -1.0)

            for j in range(1, self.nt-1):
                penalty = 1e-2 * A0  # Scales to 1 for A, Q; 1e-7 for P
                add(self.idx(j, 0, 0), self.idx(j, 0, 0), penalty/A0)
                add(self.idx(j, 0, 1), self.idx(j, 0, 1), penalty/Q0)
                add(self.idx(j, 0, 2), self.idx(j, 0, 2), penalty/P0)
                add(self.idx(j, self.nx-1, 0), self.idx(j, self.nx-1, 0), penalty/A0)
                add(self.idx(j, self.nx-1, 1), self.idx(j, self.nx-1, 1), penalty/Q0)
                add(self.idx(j, self.nx-1, 2), self.idx(j, self.nx-1, 2), penalty/P0)

            for j in range(1, self.nt - 1):
                for i in range(1, self.nx - 1):
                    aji = A[j, i] / A0
                    qji = Q[j, i] / Q0
                    pjp = P[j, i + 1] / P0
                    pjm = P[j, i - 1] / P0
                    ajp = A[j, i + 1] / A0
                    ajm = A[j, i - 1] / A0
                    qjp = Q[j, i + 1] / Q0
                    qjm = Q[j, i - 1] / Q0

                    r0 = self.idx(j, i, 0)
                    add(r0, self.idx(j + 1, i, 0),  1.0 / (2 * self.dt * 1e3))
                    add(r0, self.idx(j - 1, i, 0), -1.0 / (2 * self.dt * 1e3))
                    add(r0, self.idx(j, i + 1, 1),  Q0 / (A0 * 2 * self.dx * 1e3))
                    add(r0, self.idx(j, i - 1, 1), -Q0 / (A0 * 2 * self.dx * 1e3))

                    r1 = self.idx(j, i, 1)
                    add(r1, self.idx(j + 1, i, 1),  1.0 / (2 * self.dt * 1e3))
                    add(r1, self.idx(j - 1, i, 1), -1.0 / (2 * self.dt * 1e3))
                    add(r1, self.idx(j, i + 1, 1),  α * 2 * qjp / ajp / (2 * self.dx * 1e3))
                    add(r1, self.idx(j, i + 1, 0), -α * qjp**2 / ajp**2 / (2 * self.dx * 1e3) * (A0 / Q0))
                    add(r1, self.idx(j, i - 1, 1), -α * 2 * qjm / ajm / (2 * self.dx * 1e3))
                    add(r1, self.idx(j, i - 1, 0),  α * qjm**2 / ajm**2 / (2 * self.dx * 1e3) * (A0 / Q0))
                    add(r1, self.idx(j, i + 1, 2),  aji * A0 / (2 * self.dx * ρ * Q0 / P0 * 1e3))
                    add(r1, self.idx(j, i - 1, 2), -aji * A0 / (2 * self.dx * ρ * Q0 / P0 * 1e3))
                    dpdx = (pjp - pjm) * P0 / (2 * self.dx)
                    add(r1, self.idx(j, i, 0), dpdx * A0 / (ρ * Q0 * 1e3))
                    add(r1, self.idx(j, i, 1), KR / aji / 1e3)
                    add(r1, self.idx(j, i, 0), -KR * qji / (aji**2) * (A0 / Q0) / 1e3)

                    r2 = self.idx(j, i, 2)
                    add(r2, self.idx(j, i, 2), 1.0 / 1e3)
                    add(r2, self.idx(j, i, 0), -β / (2 * np.sqrt(aji * A0)) * (A0 / P0) / 1e3)

            J = sp.csr_matrix((data, (rows, cols)), shape=(3 * self.N, 3 * self.N))
            if np.any(np.isnan(J.data)) or np.any(np.isinf(J.data)):
                print("NaN or Inf detected in Jacobian, J min/max:", np.min(J.data), np.max(J.data))
            else:
                print("Jacobian min/max:", np.min(J.data), np.max(J.data))
            return J
        except Exception as e:
            print("Error in jacobian_crs:", e)
            raise

    def solve(self, U0, tol=1e-2, maxit=20000):
        try:
            def fun(U):
                return self.residual(U)
    
            def jac(U):
                J = self.jacobian_crs(U).toarray()
                # Precondition by column scaling
                col_norms = np.max(np.abs(J), axis=0)
                col_norms[col_norms == 0] = 1  # Avoid division by zero
                J_scaled = J / col_norms
                return J_scaled
    
            if self.bounds:
                lb, ub = self.bounds
            else:
                lb = np.full_like(U0, -np.inf)
                ub = np.full_like(U0, np.inf)
    
            tol = [tol, 0]
            parms = [20000, 20000, 1, -1, 1, 2]
            print("Starting CoDoSol with U0 shape:", U0.shape)
            result = CoDoSol(U0, fun, lb, ub, tol, parms, jac)
            if len(result) == 3:
                print("Solver failed due to infeasible initial guess")
                return None, None, None
            sol, ierr, output, history, grad, diagnostic = result
            print(f"Ierr: {ierr}, Output: {output}")
            print("Initial residual norm:", np.linalg.norm(self.residual(U0)))
            J = self.jacobian_crs(U0).toarray()
            cond_num = np.linalg.cond(J) if not np.any(np.isnan(J)) else "NaN in Jacobian"
            print("Jacobian condition number:", cond_num)
            if cond_num != "NaN in Jacobian" and cond_num > 1e8:
                print("Warning: High condition number may indicate numerical instability")
            A, Q, P = self.unpack(sol)
            return A, Q, P
        except Exception as e:
            print("Error in solve:", e)
            return None, None, None
