import numpy as np
import vtk

# --- Heart Geometry and Dynamics --- #
class Heart:
    def __init__(self, a=3.5, b=2.5, omega=2 * np.pi / 5, num_points=100):
        self.a0 = a
        self.b0 = b
        self.omega = omega
        self.num_points = num_points

    def generate_profile(self, t):
        a_t = self.a0 + 0.3 * np.cos(self.omega * t)
        b_t = self.b0 + 0.2 * np.cos(self.omega * t)
        theta = np.linspace(0, 2 * np.pi, self.num_points)
        x = a_t * np.cos(theta)
        y = b_t * np.sin(theta)
        return np.column_stack((x, y, np.zeros_like(x)))

    def split_chambers(self, polygon):
        mid_x = 0
        left = polygon[polygon[:, 0] <= mid_x]
        right = polygon[polygon[:, 0] > mid_x]
        return left, right

    def compute_area(self, polygon):
        x = polygon[:, 0]
        y = polygon[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# --- Vascular System --- #
class Vessel:
    def __init__(self):
        self.length = 1.2  # meters
        self.sections = 3  # A: ventricle, B: head, C: atrial
        self.positions = np.linspace(0, self.length, self.sections)
        self.pressure = np.array([120, 80, 10])  # mmHg
        self.velocity = np.array([0.5, 0.3, 0.1])  # m/s
        self.diameter = np.array([0.02, 0.015, 0.018])  # meters
        self.height = np.array([0.0, 0.3, -0.1])  # relative to ventricle
        self.rho = 1060  # blood density kg/m^3

    def update_hemodynamics(self):
        # Simple Bernoulli and mass conservation model
        for i in range(1, self.sections):
            delta_h = self.height[i] - self.height[i - 1]
            v1 = self.velocity[i - 1]
            A1 = np.pi * (self.diameter[i - 1] / 2) ** 2
            A2 = np.pi * (self.diameter[i] / 2) ** 2
            v2 = (A1 * v1) / A2
            dp = 0.5 * self.rho * (v2 ** 2 - v1 ** 2) + self.rho * 9.81 * delta_h
            self.velocity[i] = v2
            self.pressure[i] = self.pressure[i - 1] - dp / 133.322  # convert to mmHg


# --- Visualization --- #
def create_polydata(points):
    vtk_points = vtk.vtkPoints()
    for p in points:
        vtk_points.InsertNextPoint(p)
    lines = vtk.vtkCellArray()
    for i in range(len(points)):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, (i + 1) % len(points))
        lines.InsertNextCell(line)
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetLines(lines)
    return poly

def visualize(heart, vessel):
    renderers = [vtk.vtkRenderer() for _ in range(4)]
    render_window = vtk.vtkRenderWindow()
    interactor = vtk.vtkRenderWindowInteractor()
    render_window.SetInteractor(interactor)

    for i, r in enumerate(renderers):
        r.SetViewport(i * 0.25, 0, (i + 1) * 0.25, 1)
        r.SetBackground(0.1, 0.1, 0.2)
        render_window.AddRenderer(r)

    def update_scene(t):
        poly = create_polydata(heart.generate_profile(t))
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)
        renderers[0].RemoveAllViewProps()
        renderers[0].AddActor(actor)

        vessel.update_hemodynamics()
        bars = []
        for i in range(3):
            cylinder = vtk.vtkCylinderSource()
            cylinder.SetRadius(vessel.diameter[i] / 2)
            cylinder.SetHeight(0.1)
            cylinder.SetResolution(12)
            cylinder.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cylinder.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetPosition(i * 0.2, 0, 0)
            actor.GetProperty().SetColor(0, 1 - i * 0.5, i * 0.5)
            renderers[1].AddActor(actor)
        render_window.Render()

    def timer_callback(obj, event):
        nonlocal t
        update_scene(t)
        t += 0.1

    t = 0
    interactor.Initialize()
    interactor.AddObserver('TimerEvent', timer_callback)
    interactor.CreateRepeatingTimer(100)
    render_window.SetSize(1600, 400)
    render_window.Render()
    interactor.Start()

# --- Run Simulation --- #
heart = Heart()
vessel = Vessel()
visualize(heart, vessel)

