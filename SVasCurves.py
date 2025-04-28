import vtk

def display_curves(curve_data, renderers):
    # Clear existing actors from the renderer
    renderer = renderers[0]  # Use the first (only) renderer
    renderer.RemoveAllViewProps()

    # Colors for each curve (pressure: red, flow: green, wss: blue)
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    for i, field in enumerate(['pressure', 'flow', 'wss']):
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        scalars = vtk.vtkFloatArray()

        # Create points and lines for the curve
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(curve_data[field]))
        for j, (x, y) in enumerate(curve_data[field]):
            points.InsertNextPoint(x, y, 0)
            line.GetPointIds().SetId(j, j)
            scalars.InsertNextValue(y)
        lines.InsertNextCell(line)

        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        polydata.GetPointData().SetScalars(scalars)

        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()  # Use actor color instead of scalars

        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors[i])  # Set distinct color
        actor.GetProperty().SetLineWidth(2)

        # Add actor to the renderer
        renderer.AddActor(actor)

    renderer.ResetCamera()

def update_curves(idx, curve_data, renderers):
    # For simplicity, re-render all curves (could optimize to update only changed points)
    display_curves(curve_data, renderers)

