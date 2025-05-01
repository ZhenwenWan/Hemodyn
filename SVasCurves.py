import vtk

def display_curves(curve_values, field, point_index, point_index2, renderers):
    # Clear existing actors from the renderer
    renderer = renderers[0]  # Use the first (only) renderer
    renderer.RemoveAllViewProps()

    # Colors for the two curves (red for point_index, blue for point_index2)
    colors = [(1, 0, 0), (0, 0, 1)]

    # Plot two curves for the selected field at the two points
    for i, idx in enumerate([point_index, point_index2]):
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        scalars = vtk.vtkFloatArray()

        # Create points and lines for the curve
        line = vtk.vtkPolyLine()
        curve_data = curve_values[idx][field]
        line.GetPointIds().SetNumberOfIds(len(curve_data))
        for j, (x, y) in enumerate(curve_data):
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
        actor.GetProperty().SetColor(colors[i])  # Red for point_index, blue for point_index2
        actor.GetProperty().SetLineWidth(2)

        # Add actor to the renderer
        renderer.AddActor(actor)

    renderer.ResetCamera()

def update_curves(idx, curve_values, field, point_index, point_index2, renderers):
    # For simplicity, re-render both curves
    display_curves(curve_values, field, point_index, point_index2, renderers)

