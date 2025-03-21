import vtk
import numpy as np
import math

class Heart:
    def __init__(self):
        self.a = 4.0  # Major axis base (cm)
        self.b = 1.0  # Major axis variation (cm)
        self.omega = 2*np.pi/0.8  # HR 75 bpm
        self.minor_axis = 3.0  # Constant minor axis (cm)
        self.num_points = 100
        
    def get_shape(self, t):
        major = self.a + self.b * np.cos(self.omega * t)
        theta = np.linspace(0, 2*np.pi, self.num_points)
        x = major * np.cos(theta)
        y = self.minor_axis * np.sin(theta)
        return np.column_stack((x, y))
    
    def split_chambers(self, points):
        left = [p for p in points if p[0] <= 0]
        right = [p for p in points if p[0] > 0]
        return np.array(left), np.array(right)
    
    def calculate_area(self, points):
        x, y = points[:,0], points[:,1]
        return 0.5 * np.abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

class VascularSystem:
    def __init__(self, num_nodes=50):
        self.num_nodes = num_nodes
        self.pressure = np.zeros(num_nodes) + 80  # mmHg
        self.velocity = np.zeros(num_nodes)  # cm/s
        self.diameter = np.zeros(num_nodes) + 2.0  # cm
        self.height = np.linspace(0, 30, num_nodes)  # cm
        self.rho = 1.06  # Blood density g/cm³
        self.mu = 0.04  # Poise
        
    def update(self, heart_pressure, dt):
        # Bernoulli equation with viscous losses
        A = np.pi * (self.diameter/2)**2
        Q = self.velocity * A
        
        # Pressure gradient and viscous resistance
        delta_P = -np.diff(self.pressure)
        Re = self.rho * self.velocity[1:] * self.diameter[1:] / self.mu
        f = 64/Re  # Laminar flow
        loss = f * (self.velocity[1:]**2) * self.rho * 0.1
        
        # Update equations
        self.velocity[1:] += (delta_P - loss - self.rho*9.8*self.height[1:]/100) * dt
        self.velocity[0] = heart_pressure / (self.rho * 100)  # Simplified BC
        
        # Mass conservation
        Q[1:-1] = 0.5*(Q[:-2] + Q[2:])
        self.velocity = Q / A

class HemodynamicsVisualizer:
    def __init__(self):
        self.heart = Heart()
        self.vascular = VascularSystem()
        self.t = 0
        self.dt = 0.01
        
        # VTK Setup
        self.renderer = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.renderer)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        
        # Create 4 viewports
        self.renWin.SetSize(800, 600)
        viewports = [(0,0,0.5,0.5), (0.5,0,1,0.5),
                    (0,0.5,0.5,1), (0.5,0.5,1,1)]
        self.renderers = [self.create_viewport(vp) for vp in viewports]
        
        # Setup visualization pipelines
        self.setup_heart_visualization()
        self.setup_vascular_visualization()
        self.setup_pressure_plot()
        self.setup_velocity_plot()
        
    def create_viewport(self, vp):
        renderer = vtk.vtkRenderer()
        renderer.SetViewport(*vp)
        self.renWin.AddRenderer(renderer)
        return renderer
    
    def setup_heart_visualization(self):
        # Heart chamber visualization
        self.heart_mapper = vtk.vtkPolyDataMapper()
        self.heart_actor = vtk.vtkActor()
        self.heart_actor.SetMapper(self.heart_mapper)
        self.renderers[0].AddActor(self.heart_actor)
        
    def setup_vascular_visualization(self):
        # Vascular system tube visualization
        self.tube = vtk.vtkTubeFilter()
        self.tube.SetNumberOfSides(20)
        self.tube.SetRadius(0.3)
        
        self.vascular_mapper = vtk.vtkPolyDataMapper()
        self.vascular_actor = vtk.vtkActor()
        self.vascular_actor.SetMapper(self.vascular_mapper)
        self.renderers[1].AddActor(self.vascular_actor)
    
    def update_visualization(self):
        # Update heart shape
        points = self.heart.get_shape(self.t)
        left, right = self.heart.split_chambers(points)
        
        # Create VTK data structures
        vtk_points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        
        # Add left chamber
        for p in left:
            vtk_points.InsertNextPoint(p[0], p[1], 0)
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(left))
        for i in range(len(left)):
            polygon.GetPointIds().SetId(i, i)
        polys.InsertNextCell(polygon)
        
        # Add right chamber
        offset = len(left)
        for p in right:
            vtk_points.InsertNextPoint(p[0], p[1], 0)
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(right))
        for i in range(len(right)):
            polygon.GetPointIds().SetId(i, offset+i)
        polys.InsertNextCell(polygon)
        
        # Update heart polydata
        heart_poly = vtk.vtkPolyData()
        heart_poly.SetPoints(vtk_points)
        heart_poly.SetPolys(polys)
        self.heart_mapper.SetInputData(heart_poly)
        
        # Update vascular visualization
        # (Similar update logic for other visualization elements)
        
        self.renWin.Render()
        
    def start(self):
        self.iren.Initialize()
        self.iren.CreateRepeatingTimer(50)
        self.iren.AddObserver('TimerEvent', self.update_callback)
        self.iren.Start()
    
    def update_callback(self, obj, event):
        self.t += self.dt
        self.vascular.update(self.heart.calculate_area(), self.dt)
        self.update_visualization()

if __name__ == "__main__":
    vis = HemodynamicsVisualizer()
    vis.start()

