import numpy as np
import matplotlib.pyplot as plt
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
        self.DO_mus_zero = 0.4  # DO thredhold for motion
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
                                         (DO_val - self.DO_mus_zero) / (self.DO_mus_full - self.DO_mus_zero)
            metabolic_consumption = -metabolic_rates * DO_muscular

            DO_muscular += self.dt * (diffusive_flux_musc + exchange_flux_musc + metabolic_consumption)

            # Boundary conditions
            DO_vascular[0] = self.Oi
            DO_vascular[-1] = DO_vascular[-2]  # Neumann BC
            DO_muscular[0] = DO_muscular[1]    # Neumann BC
            DO_muscular[-1] = DO_muscular[-2]

        return self.x, DO_vascular, DO_muscular

    def _generate_Qi(self, times):
        if self.Qi_scenario == 'constant':
            return np.ones_like(times) * self.Qi_initial
        elif self.Qi_scenario == 'decrease':
            return np.linspace(self.Qi_initial, self.Qi_final, len(times))
        elif self.Qi_scenario == 'increase':
            return np.linspace(self.Qi_initial, self.Qi_final, len(times))
        else:
            raise ValueError(f"Unknown Qi scenario: {self.Qi_scenario}")


def visualize_vtk(x, DO_vascular, DO_muscular):
    """
    Visualize the final DO distribution in both compartments using VTK.
    """
    points = vtk.vtkPoints()
    for xi in x:
        points.InsertNextPoint(xi, 0, 0)

    # Create PolyLine
    lines = vtk.vtkCellArray()
    polyLine = vtk.vtkPolyLine()
    polyLine.GetPointIds().SetNumberOfIds(len(x))
    for i in range(len(x)):
        polyLine.GetPointIds().SetId(i, i)
    lines.InsertNextCell(polyLine)

    # Create PolyData
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(lines)

    # Add DO_vascular as scalar data
    DO_vasc_array = numpy_support.numpy_to_vtk(DO_vascular, deep=True)
    DO_vasc_array.SetName("DO_vascular")
    polyData.GetPointData().AddArray(DO_vasc_array)
    polyData.GetPointData().SetActiveScalars("DO_vascular")

    # Visualize
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polyData)
    mapper.SetScalarModeToUsePointData()
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)

    # Render window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    # Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

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

    x, DO_vascular, DO_muscular = model.simulate()

    # Plotting for quick check
    plt.figure(figsize=(10,5))
    plt.plot(x, DO_vascular, label='DO Vascular')
    plt.plot(x, DO_muscular, label='DO Muscular')
    plt.xlabel('Position (x)')
    plt.ylabel('DO Concentration')
    plt.legend()
    plt.title('Dissolved Oxygen Distribution')
    plt.grid(True)
    plt.show()

    # VTK visualization
    visualize_vtk(x, DO_vascular, DO_muscular)


if __name__ == "__main__":
    print("Yes")
    main()

