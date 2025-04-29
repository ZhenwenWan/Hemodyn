class ToggleButton:
    def __init__(self, text, x, y, radius, renderer, interactor, font_size=18, exec_func=None):
        self.renderer = renderer
        self.interactor = interactor
        self.toggle_state = False
        self.exec_func = exec_func

        # Create the outer circle
        self.outer_circle = self._create_circle(radius, (1, 1, 1))  # White
        self.outer_circle.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.outer_circle.SetPosition(x, y)
        self.outer_circle.GetProperty().SetDisplayLocationToForeground()

        # Create the inner circle
        self.inner_circle = self._create_circle(radius * 0.6, (1, 0, 0))  # Red
        self.inner_circle.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.inner_circle.SetPosition(x, y)
        self.inner_circle.GetProperty().SetDisplayLocationToForeground()

        # Debug: Check circle positions
        print(f"Outer circle position: {self.outer_circle.GetPosition()}")
        print(f"Inner circle position: {self.inner_circle.GetPosition()}")

        # Create the label
        self.label = vtk.vtkTextActor()
        self.label.SetInput(text)
        self.label.GetTextProperty().SetFontSize(font_size)
        self.label.GetTextProperty().SetColor(1, 1, 1)
        self.label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.label.SetPosition(x + radius * 1.5, y - radius * 0.5)

        # Add actors to the renderer
        self.renderer.AddActor2D(self.outer_circle)
        self.renderer.AddActor2D(self.inner_circle)
        self.renderer.AddActor2D(self.label)

        # Debug: Check visibility
        print(f"Outer circle visibility: {self.outer_circle.GetVisibility()}")
        print(f"Inner circle visibility: {self.inner_circle.GetVisibility()}")

        # Debug: Check props in renderer
        print(f"Number of props in renderer: {self.renderer.GetViewProps().GetNumberOfItems()}")
        for i in range(self.renderer.GetViewProps().GetNumberOfItems()):
            print(f"Prop {i}: {self.renderer.GetViewProps().GetItemAsObject(i)}")

        # Add interactivity
        self._add_interactivity()

        # Force render update
        self.renderer.GetRenderWindow().Render()

    def _create_circle(self, radius, color):
        points = vtk.vtkPoints()
        num_points = 100
        for i in range(num_points):
            angle = 2.0 * np.pi * i / num_points
            # Scale radius to normalized viewport units
            points.InsertNextPoint(radius * np.cos(angle), radius * np.sin(angle), 0)

        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(num_points)
        for i in range(num_points):
            polygon.GetPointIds().SetId(i, i)

        polygons = vtk.vtkCellArray()
        polygons.InsertNextCell(polygon)

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetPolys(polygons)

        # Use a coordinate transform to handle normalized viewport
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToNormalizedViewport()

        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(poly_data)
        mapper.SetTransformCoordinate(coord)  # Apply normalized viewport transform

        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)

        # Debug: Check geometry
        print(f"Circle created with radius {radius} and color {color}")
        print(f"Number of points: {poly_data.GetNumberOfPoints()}")
        print(f"Number of polygons: {poly_data.GetNumberOfCells()}")

        return actor

    def _add_interactivity(self):
        def on_left_click(obj, event):
            click_pos = self.interactor.GetEventPosition()
            picker = vtk.vtkPropPicker()
            picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
            if picker.GetActor2D() == self.inner_circle:
                self.toggle()

        self.interactor.AddObserver("LeftButtonPressEvent", on_left_click)

    def toggle(self):
        self.toggle_state = not self.toggle_state
        if self.toggle_state:
            self.inner_circle.GetProperty().SetColor(0, 1, 0)  # Green
        else:
            self.inner_circle.GetProperty().SetColor(1, 0, 0)  # Red

        if self.exec_func:
            self.exec_func(self.toggle_state)

        self.renderer.GetRenderWindow().Render()

