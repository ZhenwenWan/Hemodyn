import numpy as np
import vtk
from vtk.util import numpy_support


class DOModel1D:
    """
    Class to simulate 1D dissolved oxygen transport
    with vascular and muscular modules and Qi scenarios.
    """
    def __init__(self, length=10.0, nx=100, dt=0.01, total_time=1.0):
        self.length = length
        self.nx = nx
        self.dx = length / (nx - 1)
        self.dt = dt
        self.total_time = total_time
        self.x = np.linspace(0, length, nx)
        
        # Default parameters
        self.D_vascular = 0.1
        self.D_muscular = 0.05
        self.exchange_rate = 0.01
        self.metabolism_rate = 0.02
        self.Oi = 1.0  # Inlet DO concentration
        self.DO_mus_zero = 0.4  # DO threshold for motion
        self.DO_mus_full = 0.8  # DO for full motion
        
        # Qi scenarios
        self.Qi_scenario = 'constant'
        self.Qi_initial = 1.0
        self.Qi_final = 0.5

    def set_conditions(self, D_vascular=None, D_muscular=None, exchange_rate=None, metabolism_rate=None,
                       Oi=None, Qi_scenario=None, Qi_initial=None, Qi_final=None):
        if D_vascular is not None:
            self.D_vascular = D_vascular
        if D_muscular is not None:
            self.D_muscular = D_muscular
        if exchange_rate is not None:
            self.exchange_rate = exchange_rate
        if metabolism_rate is not None:
            self.metabolism_rate = metabolism_rate
        if Oi is not None:
            self.Oi = Oi
        if Qi_scenario is not None:
            self.Qi_scenario = Qi_scenario
        if Qi_initial is not None:
            self.Qi_initial = Qi_initial
        if Qi_final is not None:
            self.Qi_final = Qi_final

    def simulate(self):
        # Initialize concentrations
        DO_vascular = np.ones(self.nx) * self.Oi
        DO_muscular = np.ones(self.nx) * self.Oi
        
        times = np.arange(0, self.total_time, self.dt)
        Qi_values = self._generate_Qi(times)
        
        # Store results for all time steps
        DO_vascular_all = np.zeros((len(times), self.nx))
        DO_muscular_all = np.zeros((len(times), self.nx))
        
        for t_idx, t in enumerate(times):
            Qi = Qi_values[t_idx]
            
            # Vascular advection-diffusion + exchange
            advective_flux = -Qi * np.gradient(DO_vascular, self.dx)
            diffusive_flux = self.D_vascular * np.gradient(np.gradient(DO_vascular, self.dx), self.dx)
            exchange_flux_vasc = -self.exchange_rate * (DO_vascular - DO_muscular)

            DO_vascular += self.dt * (advective_flux + diffusive_flux + exchange_flux_vasc)
            
            # Muscular diffusion + exchange + metabolism
            diffusive_flux_musc = self.D_muscular * np.gradient(np.gradient(DO_muscular, self.dx), self.dx)
            exchange_flux_musc = self.exchange_rate * (DO_vascular - DO_muscular)
            # Nonlinear metabolic consumption term
            metabolic_rates = np.zeros(self.nx)
            for i in range(self.nx):
                DO_val = DO_muscular[i]
                if DO_val >= self.DO_mus_full:
                    metabolic_rates[i] = self.metabolism_rate
                elif DO_val <= self.DO_mus_zero:
                    metabolic_rates[i] = 0.0
                else:
                    metabolic_rates[i] = self.metabolism_rate * \
                                         (DO_val - self.DO_mus_zero) / (DO_mus_full - self.DO_mus_zero)
            metabolic_consumption = -metabolic_rates * DO_muscular

            DO_muscular += self.dt * (diffusive_flux_musc + exchange_flux_musc + metabolic_consumption)

            # Boundary conditions
            DO_vascular[0] = self.Oi
            DO_vascular[-1] = DO_vascular[-2]  # Neumann BC
            DO_muscular[0] = DO_muscular[1]    # Neumann BC
            DO_muscular[-1] = DO_muscular[-2]
            
            # Store results
            DO_vascular_all[t_idx, :] = DO_vascular.copy()
            DO_muscular_all[t_idx, :] = DO_muscular.copy()

        return self.x, times, DO_vascular_all, DO_muscular_all

    def _generate_Qi(self, times):
        if self.Qi_scenario == 'constant':
            return np.ones_like(times) * self.Qi_initial
        elif self.Qi_scenario == 'decrease':
            return np.linspace(self.Qi_initial, self.Qi_final, len(times))
        elif self.Qi_scenario == 'increase':
            return np.linspace(self.Qi_initial, self.Qi_final, len(times))
        else:
            raise ValueError(f"Unknown Qi scenario: {self.Qi_scenario}")


def visualize_vtk(x, times, DO_vascular_all, DO_muscular_all):
    """
    Visualize the DO distribution in four renderers:
    - Renderer 0: 2D DO_vascular (x vs time)
    - Renderer 1: 2D DO_muscular (x vs time)
    - Renderer 2: 1D DO_vascular vs DO_muscular (last step)
    - Renderer 3: 1D DO_vascular vs DO_muscular at x=nx/2 over time
    """
    # Initialize render window and renderers
    renderWindow = vtk.vtkRenderWindow()
    renderers = [vtk.vtkRenderer() for _ in range(4)]
    
    # Set viewport for each renderer (2x2 grid)
    renderers[0].SetViewport(0.0, 0.5, 0.5, 1.0)  # Top-left
    renderers[1].SetViewport(0.5, 0.5, 1.0, 1.0)  # Top-right
    renderers[2].SetViewport(0.0, 0.0, 0.5, 0.5)  # Bottom-left
    renderers[3].SetViewport(0.5, 0.0, 1.0, 0.5)  # Bottom-right
    
    for renderer in renderers:
        renderWindow.AddRenderer(renderer)
        renderer.SetBackground(0.1, 0.2, 0.3)

    # Renderer 0: 2D DO_vascular (x vs time)
    points_vasc = vtk.vtkPoints()
    for t_idx, t in enumerate(times):
        for x_idx, xi in enumerate(x):
            points_vasc.InsertNextPoint(xi, t, DO_vascular_all[t_idx, x_idx])
    
    grid_vasc = vtk.vtkStructuredGrid()
    grid_vasc.SetDimensions(len(x), len(times), 1)
    grid_vasc.SetPoints(points_vasc)
    
    mapper_vasc = vtk.vtkDataSetMapper()
    mapper_vasc.SetInputData(grid_vasc)
    mapper_vasc.ScalarVisibilityOff()
    
    actor_vasc = vtk.vtkActor()
    actor_vasc.SetMapper(mapper_vasc)
    renderers[0].AddActor(actor_vasc)

    # Renderer 1: 2D DO_muscular (x vs time)
    points_musc = vtk.vtkPoints()
    for t_idx, t in enumerate(times):
        for x_idx, xi in enumerate(x):
            points_musc.InsertNextPoint(xi, t, DO_muscular_all[t_idx, x_idx])
    
    grid_musc = vtk.vtkStructuredGrid()
    grid_musc.SetDimensions(len(x), len(times), 1)
    grid_musc.SetPoints(points_musc)
    
    mapper_musc = vtk.vtkDataSetMapper()
    mapper_musc.SetInputData(grid_musc)
    mapper_musc.ScalarVisibilityOff()
    
    actor_musc = vtk.vtkActor()
    actor_musc.SetMapper(mapper_musc)
    renderers[1].AddActor(actor_musc)

    # Renderer 2: 1D DO_vascular vs DO_muscular (last step)
    points_last = vtk.vtkPoints()
    for x_idx in range(len(x)):
        points_last.InsertNextPoint(DO_vascular_all[-1, x_idx], DO_muscular_all[-1, x_idx], 0)
    
    lines_last = vtk.vtkCellArray()
    polyLine_last = vtk.vtkPolyLine()
    polyLine_last.GetPointIds().SetNumberOfIds(len(x))
    for i in range(len(x)):
        polyLine_last.GetPointIds().SetId(i, i)
    lines_last.InsertNextCell(polyLine_last)
    
    polyData_last = vtk.vtkPolyData()
    polyData_last.SetPoints(points_last)
    polyData_last.SetLines(lines_last)
    
    mapper_last = vtk.vtkPolyDataMapper()
    mapper_last.SetInputData(polyData_last)
    
    actor_last = vtk.vtkActor()
    actor_last.SetMapper(mapper_last)
    renderers[2].AddActor(actor_last)

    # Renderer 3: 1D DO_vascular vs DO_muscular at x=nx/2 over time
    mid_idx = len(x) // 2
    points_mid = vtk.vtkPoints()
    for t_idx, t in enumerate(times):
        points_mid.InsertNextPoint(DO_vascular_all[t_idx, mid_idx], DO_muscular_all[t_idx, mid_idx], 0)
    
    lines_mid = vtk.vtkCellArray()
    polyLine_mid = vtk.vtkPolyLine()
    polyLine_mid.GetPointIds().SetNumberOfIds(len(times))
    for i in range(len(times)):
        polyLine_mid.GetPointIds().SetId(i, i)
    lines_mid.InsertNextCell(polyLine_mid)
    
    polyData_mid = vtk.vtkPolyData()
    polyData_mid.SetPoints(points_mid)
    polyData_mid.SetLines(lines_mid)
    
    mapper_mid = vtk.vtkPolyDataMapper()
    mapper_mid.SetInputData(polyData_mid)
    
    actor_mid = vtk.vtkActor()
    actor_mid.SetMapper(mapper_mid)
    renderers[3].AddActor(actor_mid)

    # Setup interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    
    # Adjust camera for each renderer
    for renderer in renderers:
        renderer.ResetCamera()
    
    renderWindow.Render()
    renderWindowInteractor.Start()


def main():
    # User-friendly configuration
    model = DOModel1D(length=10.0, nx=200, dt=0.01, total_time=5.0)
    model.set_conditions(
        D_vascular=0.1,
        D_muscular=0.05,
        exchange_rate=0.01,
        metabolism_rate=0.02,
        Oi=1.0,
        Qi_scenario='decrease',  # Options: 'constant', 'decrease', 'increase'
        Qi_initial=1.0,
        Qi_final=0.5
    )

    x, times, DO_vascular_all, DO_muscular_all = model.simulate()

    # VTK visualization
    visualize_vtk(x, times, DO_vascular_all, DO_muscular_all)


if __name__ == "__main__":
    print("Yes")
    main()

