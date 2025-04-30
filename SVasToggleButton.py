import vtk
import numpy as np

class ToggleButton:
    def __init__(self, text, x, y, radius, renderers, interactor, font_size=18):
        self.renderers = renderers
        self.interactor = interactor
        self.toggle_state = False
        self.x = x
        self.y = y
        self.radius = radius

        # Outer circle (white)
        self.outer_circle = self._create_circle(radius, (1, 1, 1))
        self.outer_circle.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.outer_circle.SetPosition(x, y)

        # Inner circle (red/green)
        self.inner_circle = self._create_circle(radius * 0.6, (1, 0, 0))
        self.inner_circle.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.inner_circle.SetPosition(x, y)

        # Label
        self.label = vtk.vtkTextActor()
        self.label.SetInput(text)
        self.label.GetTextProperty().SetFontSize(font_size)
        self.label.GetTextProperty().SetColor(1, 1, 1)
        self.label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.label.SetPosition(x + radius * 1.5, y - radius * 0.5)

        # Add to renderers[0]
        renderer = self.renderers[0]
        renderer.AddActor2D(self.outer_circle)
        renderer.AddActor2D(self.inner_circle)
        renderer.AddActor2D(self.label)

        self._add_interactivity()
        self.interactor.GetRenderWindow().Render()

    def _create_circle(self, radius, color):
        circle_source = vtk.vtkRegularPolygonSource()
        circle_source.SetNumberOfSides(100)
        circle_source.SetRadius(radius)
        circle_source.SetCenter(0, 0, 0)
        circle_source.Update()

        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputConnection(circle_source.GetOutputPort())
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToNormalizedViewport()
        mapper.SetTransformCoordinate(coord)

        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        return actor

    def _add_interactivity(self):
        def on_left_click(obj, event):
            click_pos = self.interactor.GetEventPosition()
            ren_win = self.interactor.GetRenderWindow()
            window_size = ren_win.GetSize()
            norm_x = click_pos[0] / window_size[0]
            norm_y = click_pos[1] / window_size[1]
            # Increase click area slightly for better usability
            if (norm_x - self.x) ** 2 + (norm_y - self.y) ** 2 <= (self.radius * 1.5) ** 2:
                self.toggle()
        self.interactor.AddObserver("LeftButtonPressEvent", on_left_click)

    def toggle(self):
        self.toggle_state = not self.toggle_state
        if self.toggle_state:
            self.inner_circle.GetProperty().SetColor(0, 1, 0)  # Green
            self.renderers[0].ResetCamera()  # Resets shared camera for renderers[:3]
        else:
            self.inner_circle.GetProperty().SetColor(1, 0, 0)  # Red
        self.interactor.GetRenderWindow().Render()

