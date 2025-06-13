import numpy as np
import vtk
from vtk.util import numpy_support


class DOModel1D:
    """
    Class to simulate 1D dissolved oxygen transport
    with vascular and muscular modules and Qi scenarios.
    """
    def __init__(self, length=10.0, nx=100, dt=0.01, total_time=10.0):
        self.length = length
        self.nx = nx
        self.dx = length / (nx - 1)
        self.dt = dt
        self.total_time = total_time
        self.x = np.linspace(0, length, nx)
        
        # Default parameters
        self.D_vascular = 0.1
        self.D_muscular = 0.05
        self.exchange_rate = 0.5
        self.metabolism_rate = 0.5
        self.Oi = 1.0
        self.DO_mus_zero = 0.4
        self.DO_mus_full = 0.8
        
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
        DO_vascular = np.ones(self.nx) * self.Oi
        DO_muscular = np.ones(self.nx) * self.Oi
        
        times = np.arange(0, self.total_time, self.dt)
        Qi_values = self._generate_Qi(times)
        
        DO_vascular_all = np.zeros((len(times), self.nx))
        DO_muscular_all = np.zeros((len(times), self.nx))
        
        for t_idx, t in enumerate(times):
            Qi = Qi_values[t_idx]
            
            advective_flux = -Qi * np.gradient(DO_vascular, self.dx)
            diffusive_flux = self.D_vascular * np.gradient(np.gradient(DO_vascular, self.dx), self.dx)
            exchange_flux_vasc = -self.exchange_rate * (DO_vascular - DO_muscular)
            DO_vascular += self.dt * (advective_flux + diffusive_flux + exchange_flux_vasc)
            
            diffusive_flux_musc = self.D_muscular * np.gradient(np.gradient(DO_muscular, self.dx), self.dx)
            exchange_flux_musc = self.exchange_rate * (DO_vascular - DO_muscular)
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

            DO_vascular[0] = self.Oi
            DO_vascular[-1] = DO_vascular[-2]
            DO_muscular[0] = DO_muscular[1]
            DO_muscular[-1] = DO_muscular[-2]
            
            DO_vascular_all[t_idx, :] = DO_vascular.copy()
            DO_muscular_all[t_idx, :] = DO_muscular.copy()

        print(f"DO_vascular_all min: {np.min(DO_vascular_all)}, max: {np.max(DO_vascular_all)}")
        print(f"DO_muscular_all min: {np.min(DO_muscular_all)}, max: {np.max(DO_muscular_all)}")
        
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
    - Renderer 2: 1D DO_vascular and DO_muscular vs x (last step)
    - Renderer 3: 1D DO_vascular and DO_muscular vs time at x=nx/2
    """
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(1200, 800)  # Adjust width and height as needed
    renderers = [vtk.vtkRenderer() for _ in range(4)]
    
    renderers[0].SetViewport(0.0, 0.5, 0.5, 1.0)
    renderers[1].SetViewport(0.5, 0.5, 1.0, 1.0)
    renderers[2].SetViewport(0.0, 0.0, 0.5, 0.5)
    renderers[3].SetViewport(0.5, 0.0, 1.0, 0.5)
    
    # Differentiated background colors
    renderers[0].SetBackground(0.1, 0.2, 0.3)  # Dark blue
    renderers[1].SetBackground(0.15, 0.25, 0.3)  # Slightly lighter blue
    renderers[2].SetBackground(0.1, 0.25, 0.25)  # Teal
    renderers[3].SetBackground(0.15, 0.2, 0.25)  # Dark teal
    for renderer in renderers:
        renderWindow.AddRenderer(renderer)

    # Separate lookup tables
    lut_vasc = vtk.vtkLookupTable()
    lut_vasc.SetHueRange(0.667, 0.0)  # Blue (low) to red (high)
    lut_vasc.SetRange(np.min(DO_vascular_all), np.max(DO_vascular_all))
    lut_vasc.Build()

    lut_musc = vtk.vtkLookupTable()
    lut_musc.SetHueRange(0.667, 0.0)  # Blue (low) to red (high)
    lut_musc.SetRange(np.min(DO_muscular_all), np.max(DO_muscular_all))
    lut_musc.Build()

    # Renderer 0: 2D DO_vascular (x vs time)
    points_vasc = vtk.vtkPoints()
    scalars_vasc = np.ravel(DO_vascular_all, order='C')
    for t_idx, t in enumerate(times):
        for x_idx, xi in enumerate(x):
            points_vasc.InsertNextPoint(xi, t, 0)
    
    polyData_vasc = vtk.vtkPolyData()
    polyData_vasc.SetPoints(points_vasc)
    
    scalar_array_vasc = numpy_support.numpy_to_vtk(scalars_vasc, deep=True)
    scalar_array_vasc.SetName("DO_vascular")
    polyData_vasc.GetPointData().SetScalars(scalar_array_vasc)
    
    # Create triangles for smooth surface
    triangles_vasc = vtk.vtkCellArray()
    for t_idx in range(len(times)-1):
        for x_idx in range(len(x)-1):
            triangle1 = vtk.vtkTriangle()
            triangle1.GetPointIds().SetId(0, t_idx * len(x) + x_idx)
            triangle1.GetPointIds().SetId(1, t_idx * len(x) + x_idx + 1)
            triangle1.GetPointIds().SetId(2, (t_idx + 1) * len(x) + x_idx)
            triangles_vasc.InsertNextCell(triangle1)
            
            triangle2 = vtk.vtkTriangle()
            triangle2.GetPointIds().SetId(0, t_idx * len(x) + x_idx + 1)
            triangle2.GetPointIds().SetId(1, (t_idx + 1) * len(x) + x_idx + 1)
            triangle2.GetPointIds().SetId(2, (t_idx + 1) * len(x) + x_idx)
            triangles_vasc.InsertNextCell(triangle2)
    
    polyData_vasc.SetPolys(triangles_vasc)
    
    mapper_vasc = vtk.vtkPolyDataMapper()
    mapper_vasc.SetInputData(polyData_vasc)
    mapper_vasc.SetScalarModeToUsePointData()
    mapper_vasc.SetLookupTable(lut_vasc)
    mapper_vasc.ScalarVisibilityOn()
    
    actor_vasc = vtk.vtkActor()
    actor_vasc.SetMapper(mapper_vasc)
    renderers[0].AddActor(actor_vasc)

    # Color bar for Renderer 0
    scalar_bar_vasc = vtk.vtkScalarBarActor()
    scalar_bar_vasc.SetLookupTable(lut_vasc)
    scalar_bar_vasc.SetTitle("DO_vascular")
    scalar_bar_vasc.SetNumberOfLabels(5)
    scalar_bar_vasc.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    scalar_bar_vasc.GetPositionCoordinate().SetValue(0.1, 0.05)
    scalar_bar_vasc.SetWidth(0.1)
    scalar_bar_vasc.SetHeight(0.4)
    renderers[0].AddActor2D(scalar_bar_vasc)

    # Renderer 1: 2D DO_muscular (x vs time)
    points_musc = vtk.vtkPoints()
    scalars_musc = np.ravel(DO_muscular_all, order='C')
    for t_idx, t in enumerate(times):
        for x_idx, xi in enumerate(x):
            points_musc.InsertNextPoint(xi, t, 0)
    
    polyData_musc = vtk.vtkPolyData()
    polyData_musc.SetPoints(points_musc)
    
    scalar_array_musc = numpy_support.numpy_to_vtk(scalars_musc, deep=True)
    scalar_array_musc.SetName("DO_muscular")
    polyData_musc.GetPointData().SetScalars(scalar_array_musc)
    
    triangles_musc = vtk.vtkCellArray()
    for t_idx in range(len(times)-1):
        for x_idx in range(len(x)-1):
            triangle1 = vtk.vtkTriangle()
            triangle1.GetPointIds().SetId(0, t_idx * len(x) + x_idx)
            triangle1.GetPointIds().SetId(1, t_idx * len(x) + x_idx + 1)
            triangle1.GetPointIds().SetId(2, (t_idx + 1) * len(x) + x_idx)
            triangles_musc.InsertNextCell(triangle1)
            
            triangle2 = vtk.vtkTriangle()
            triangle2.GetPointIds().SetId(0, t_idx * len(x) + x_idx + 1)
            triangle2.GetPointIds().SetId(1, (t_idx + 1) * len(x) + x_idx + 1)
            triangle2.GetPointIds().SetId(2, (t_idx + 1) * len(x) + x_idx)
            triangles_musc.InsertNextCell(triangle2)
    
    polyData_musc.SetPolys(triangles_musc)
    
    mapper_musc = vtk.vtkPolyDataMapper()
    mapper_musc.SetInputData(polyData_musc)
    mapper_musc.SetScalarModeToUsePointData()
    mapper_musc.SetLookupTable(lut_musc)
    mapper_musc.ScalarVisibilityOn()
    
    actor_musc = vtk.vtkActor()
    actor_musc.SetMapper(mapper_musc)
    renderers[1].AddActor(actor_musc)

    # Color bar for Renderer 1
    scalar_bar_musc = vtk.vtkScalarBarActor()
    scalar_bar_musc.SetLookupTable(lut_musc)
    scalar_bar_musc.SetTitle("DO_muscular")
    scalar_bar_musc.SetNumberOfLabels(5)
    scalar_bar_musc.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    scalar_bar_musc.GetPositionCoordinate().SetValue(0.1, 0.05)
    scalar_bar_musc.SetWidth(0.1)
    scalar_bar_musc.SetHeight(0.4)
    renderers[1].AddActor2D(scalar_bar_musc)

    # Renderer 2: 1D DO_vascular and DO_muscular vs x (last step)
    points_vasc_last = vtk.vtkPoints()
    for x_idx, xi in enumerate(x):
        points_vasc_last.InsertNextPoint(xi, DO_vascular_all[-1, x_idx], 0)
    
    lines_vasc_last = vtk.vtkCellArray()
    polyLine_vasc_last = vtk.vtkPolyLine()
    polyLine_vasc_last.GetPointIds().SetNumberOfIds(len(x))
    for i in range(len(x)):
        polyLine_vasc_last.GetPointIds().SetId(i, i)
    lines_vasc_last.InsertNextCell(polyLine_vasc_last)
    
    polyData_vasc_last = vtk.vtkPolyData()
    polyData_vasc_last.SetPoints(points_vasc_last)
    polyData_vasc_last.SetLines(lines_vasc_last)
    
    mapper_vasc_last = vtk.vtkPolyDataMapper()
    mapper_vasc_last.SetInputData(polyData_vasc_last)
    
    actor_vasc_last = vtk.vtkActor()
    actor_vasc_last.SetMapper(mapper_vasc_last)
    actor_vasc_last.GetProperty().SetColor(1, 0, 0)
    renderers[2].AddActor(actor_vasc_last)

    points_musc_last = vtk.vtkPoints()
    for x_idx, xi in enumerate(x):
        points_musc_last.InsertNextPoint(xi, DO_muscular_all[-1, x_idx], 0)
    
    lines_musc_last = vtk.vtkCellArray()
    polyLine_musc_last = vtk.vtkPolyLine()
    polyLine_musc_last.GetPointIds().SetNumberOfIds(len(x))
    for i in range(len(x)):
        polyLine_musc_last.GetPointIds().SetId(i, i)
    lines_musc_last.InsertNextCell(polyLine_musc_last)
    
    polyData_musc_last = vtk.vtkPolyData()
    polyData_musc_last.SetPoints(points_musc_last)
    polyData_musc_last.SetLines(lines_musc_last)
    
    mapper_musc_last = vtk.vtkPolyDataMapper()
    mapper_musc_last.SetInputData(polyData_musc_last)
    
    actor_musc_last = vtk.vtkActor()
    actor_musc_last.SetMapper(mapper_musc_last)
    actor_musc_last.GetProperty().SetColor(0, 0, 1)
    renderers[2].AddActor(actor_musc_last)

    axes_actor2 = vtk.vtkAxesActor()
    axes_actor2.SetTotalLength(5, 0.5, 0)
    axes_actor2.SetXAxisLabelText("x")
    axes_actor2.SetYAxisLabelText("DO")
    axes_actor2.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1, 1, 1)
    axes_actor2.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1, 1, 1)
    renderers[2].AddActor(axes_actor2)

    # Renderer 3: 1D DO_vascular and DO_muscular vs time at x=nx/2
    mid_idx = len(x) // 2
    points_vasc_mid = vtk.vtkPoints()
    for t_idx, t in enumerate(times):
        points_vasc_mid.InsertNextPoint(t, DO_vascular_all[t_idx, mid_idx], 0)
    
    lines_vasc_mid = vtk.vtkCellArray()
    polyLine_vasc_mid = vtk.vtkPolyLine()
    polyLine_vasc_mid.GetPointIds().SetNumberOfIds(len(times))
    for i in range(len(times)):
        polyLine_vasc_mid.GetPointIds().SetId(i, i)
    lines_vasc_mid.InsertNextCell(polyLine_vasc_mid)
    
    polyData_vasc_mid = vtk.vtkPolyData()
    polyData_vasc_mid.SetPoints(points_vasc_mid)
    polyData_vasc_mid.SetLines(lines_vasc_mid)
    
    mapper_vasc_mid = vtk.vtkPolyDataMapper()
    mapper_vasc_mid.SetInputData(polyData_vasc_mid)
    
    actor_vasc_mid = vtk.vtkActor()
    actor_vasc_mid.SetMapper(mapper_vasc_mid)
    actor_vasc_mid.GetProperty().SetColor(1, 0, 0)
    renderers[3].AddActor(actor_vasc_mid)

    points_musc_mid = vtk.vtkPoints()
    for t_idx, t in enumerate(times):
        points_musc_mid.InsertNextPoint(t, DO_muscular_all[t_idx, mid_idx], 0)
    
    lines_musc_mid = vtk.vtkCellArray()
    polyLine_musc_mid = vtk.vtkPolyLine()
    polyLine_musc_mid.GetPointIds().SetNumberOfIds(len(times))
    for i in range(len(times)):
        polyLine_musc_mid.GetPointIds().SetId(i, i)
    lines_musc_mid.InsertNextCell(polyLine_musc_mid)
    
    polyData_musc_mid = vtk.vtkPolyData()
    polyData_musc_mid.SetPoints(points_musc_mid)
    polyData_musc_mid.SetLines(lines_musc_mid)
    
    mapper_musc_mid = vtk.vtkPolyDataMapper()
    mapper_musc_mid.SetInputData(polyData_musc_mid)
    
    actor_musc_mid = vtk.vtkActor()
    actor_musc_mid.SetMapper(mapper_musc_mid)
    actor_musc_mid.GetProperty().SetColor(0, 0, 1)
    renderers[3].AddActor(actor_musc_mid)

    axes_actor3 = vtk.vtkAxesActor()
    axes_actor3.SetTotalLength(5, 0.5, 0)
    axes_actor3.SetXAxisLabelText("Time")
    axes_actor3.SetYAxisLabelText("DO")
    axes_actor3.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1, 1, 1)
    axes_actor3.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1, 1, 1)
    renderers[3].AddActor(axes_actor3)

    # Setup camera for each renderer
    for i, renderer in enumerate(renderers):
        camera = renderer.GetActiveCamera()
        if i < 2:
            camera.SetPosition(np.mean(x), np.mean(times), 10)
            camera.SetFocalPoint(np.mean(x), np.mean(times), 0)
            camera.SetViewUp(0, 1, 0)
        else:
            camera.SetPosition(5, 0.5, 10)
            camera.SetFocalPoint(5, 0.5, 0)
            camera.SetViewUp(0, 1, 0)
        renderer.ResetCameraClippingRange()

    renderWindow.Render()
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Start()


def main():
    model = DOModel1D(length=10.0, nx=200, dt=0.01, total_time=10.0)
    model.set_conditions(
        D_vascular=0.01,
        D_muscular=0.05,
        exchange_rate=0.25,
        metabolism_rate=0.15,
        Oi=1.0,
        Qi_scenario='decrease',
        Qi_initial=1.0,
        Qi_final=0.5
    )

    x, times, DO_vascular_all, DO_muscular_all = model.simulate()
    visualize_vtk(x, times, DO_vascular_all, DO_muscular_all)


if __name__ == "__main__":
    print("Yes")
    main()
