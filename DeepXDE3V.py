
# DeepXDE3V_synth.py  (fixed Neumann at downstream boundary)
import numpy as np
import deepxde as dde
import viz

def linspace_2d_grid(xmin, xmax, Nx, tmin, tmax, Nt):
    x = np.linspace(xmin, xmax, Nx); t = np.linspace(tmin, tmax, Nt)
    X, T = np.meshgrid(x, t, indexing="xy"); return x, t, X, T

def synth_oxygen_ADR(L=0.2, T=1.0, Nx=121, Nt=2500, A0=3.14e-5, u0=0.3, u_amp=0.25, D_true=1.2e-6, k_c_true=0.08):
    import scipy.sparse as sp, scipy.sparse.linalg as spla
    x = np.linspace(0.0, L, Nx); t = np.linspace(0.0, T, Nt)
    dx = x[1]-x[0]; dt = t[1]-t[0]
    alpha = dt*D_true/dx**2

    def u_xt(xv,tv): return u0*(1.0 + u_amp*np.sin(2*np.pi*tv/T - 2*np.pi*xv/L))
    def O_in_func(tv): return 0.06 + 0.02*np.sin(2*np.pi*tv/T)

    O = np.zeros((Nt,Nx)); O[0,:] = O_in_func(0.0)

    # Build constant matrix for BE on diffusion+reaction:
    # interior rows:  (1 + dt*k_c + 2*alpha) * O_i  - alpha*(O_{i-1}+O_{i+1})
    main = (1.0 + dt*k_c_true + 2.0*alpha) * np.ones(Nx)
    off  = (- alpha) * np.ones(Nx - 1)
    A = sp.diags([off, main, off], offsets=[-1,0,1], shape=(Nx,Nx), format="lil")

    # Left boundary (Dirichlet) row: identity; RHS gets O_in(t_{n+1})
    A[0, :] = 0.0; A[0, 0] = 1.0

    # Right boundary (Neumann O_x=0) with **second-order** ghost elimination:
    # O_N = O_{N-2}  => O_xx(N-1) ≈ 2*(O_{N-2} - O_{N-1})/dx^2
    # row: (1 + dt*k_c + 2*alpha)*O_{N-1}  - 2*alpha*O_{N-2}
    A[-1, :] = 0.0
    A[-1, -1] = 1.0 + dt*k_c_true + 2.0*alpha
    A[-1, -2] = -2.0*alpha
    A = A.tocsc()

    for n in range(Nt-1):
        tn = t[n+1]
        rhs = O[n, :].copy()

        # Explicit upwind advection at time level n
        u_now = u_xt(x, t[n]*np.ones_like(x))
        O_x_explicit = np.zeros_like(O[n, :])
        for i in range(1,Nx-1):
            O_x_explicit[i] = (O[n,i]-O[n,i-1])/dx if u_now[i]>=0 else (O[n,i+1]-O[n,i])/dx
        O_x_explicit[0]  = 0.0 if u_now[0]>=0 else (O[n,1]-O[n,0])/dx
        O_x_explicit[-1] = (O[n,-1]-O[n,-2])/dx if u_now[-1]>=0 else 0.0

        rhs = rhs - dt*(u_now*O_x_explicit)

        # Dirichlet at x=0 goes to RHS
        rhs[0] = O_in_func(tn)

        # Solve BE system
        O[n+1, :] = spla.spsolve(A, rhs)

    X,Tgrid = np.meshgrid(x,t, indexing="xy"); U = u_xt(X,Tgrid); A_true = A0*np.ones_like(U); Q_true=A_true*U
    O_up, O_down = O[:,0], O[:,-1]
    return x,t,A_true,Q_true,O,O_up,O_down

def train_pinn_with_two_timeseries(L=0.2, T=1.0, D_init=1.0e-6, k_c_init=0.05, num_domain=3000, num_boundary=300, num_initial=100, iters_adam=15000):
    x,t,A_true,Q_true,O_true,O_up,O_down = synth_oxygen_ADR(L=L, T=T)
    A0 = A_true[0,0]
    def u_xt_numpy(xv,tv,u0=0.3,u_amp=0.25): b = dde.backend; return u0*(1.0 + u_amp*b.sin(2*np.pi*tv/T - 2*np.pi*xv/L))
    t_arr = t.copy()
    def O_in_func_batch(X): tt = X[:,1]; return np.interp(tt, t_arr, O_up).reshape(-1,1)
    def O_out_timeseries_points(): X_meas = np.column_stack([np.full_like(t_arr,L), t_arr]); Y_meas = O_down.reshape(-1,1); return X_meas, Y_meas
    geom= dde.geometry.Interval(0.0,L); timed = dde.geometry.TimeDomain(0.0,T); geomtime = dde.geometry.GeometryXTime(geom,timed)
    D_var = dde.Variable(D_init); kc_var = dde.Variable(k_c_init)
    def pde_O_only(X,y):
        O = y[:,2:3]; O_t = dde.grad.jacobian(y,X,i=2,j=1); O_x = dde.grad.jacobian(y,X,i=2,j=0); O_xx = dde.grad.hessian(y,X,component=2,i=0,j=0)
        u_vals = u_xt_numpy(X[:,0:1], X[:,1:2]); return [O_t + u_vals*O_x - D_var*O_xx + kc_var*O]
    def on_ic(_,on_initial): return on_initial
    ic_A = dde.icbc.IC(geomtime, lambda X: A0*np.ones_like(X[:,:1]), on_ic, component=0)
    ic_Q = dde.icbc.IC(geomtime, lambda X: 0.0*np.ones_like(X[:,:1]), on_ic, component=1)
    ic_O = dde.icbc.IC(geomtime, lambda X: np.full_like(X[:,:1], O_up[0]), on_ic, component=2)
    def is_left(X,on_b): return on_b and dde.utils.isclose(X[0],0.0)
    bc_O_in = dde.icbc.DirichletBC(geomtime, O_in_func_batch, is_left, component=2)
    X_meas, Y_meas = O_out_timeseries_points(); bc_O_out_data = dde.icbc.PointSetBC(X_meas, Y_meas, component=2)
    data = dde.data.TimePDE(geomtime, pde_O_only, [ic_A,ic_Q,ic_O,bc_O_in,bc_O_out_data], num_domain=num_domain, num_boundary=num_boundary, num_initial=num_initial)
    net = dde.nn.FNN([2]+[64]*3+[3], "tanh", "Glorot uniform"); model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, external_trainable_variables=[D_var,kc_var]); model.train(iterations=iters_adam)
    model.compile("L-BFGS", external_trainable_variables=[D_var,kc_var]); model.train()
    D_learned, kc_learned = model.sess.run([D_var, kc_var])
    print("Learned parameters: D =", float(D_learned), "  k_c =", float(kc_learned))
    np.savez_compressed("synthetic_artery_adr.npz", x=x,t=t,A_true=A_true,Q_true=Q_true,O_true=O_true,O_up=O_up,O_down=O_down)

if __name__=="__main__":
    train_pinn_with_two_timeseries()
