import vtk

def display_curves(curve_values, field, point_index, point_index2, renderer):
    try:
        # Clear all existing actors and reset renderer state
        renderer.RemoveAllViewProps()
        renderer.Clear()  # Explicitly clear renderer
        colors = [(1, 0, 0), (0, 0, 1)]

        # Validate indices and plot curves
        for i, idx in enumerate([point_index, point_index2]):
            # Check if index is valid
            if idx < 0 or idx >= len(curve_values):
                print(f"Invalid index {idx} in display_curves, max: {len(curve_values)-1}")
                continue
            # Check if field data exists and is non-empty
            if field not in curve_values[idx] or not curve_values[idx][field]:
                print(f"No data for field {field} at index {idx}")
                continue

            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            scalars = vtk.vtkFloatArray()
            line = vtk.vtkPolyLine()
            curve_data = curve_values[idx][field]

            # Validate curve data
            if not curve_data or len(curve_data) < 2:
                print(f"Insufficient curve data for index {idx}, field {field}")
                continue

            line.GetPointIds().SetNumberOfIds(len(curve_data))
            for j, (x, y) in enumerate(curve_data):
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    print(f"Invalid data point at index {idx}, field {field}, position {j}")
                    continue
                points.InsertNextPoint(x, y, 0)
                line.GetPointIds().SetId(j, j)
                scalars.InsertNextValue(y)
            lines.InsertNextCell(line)

            # Create polydata
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetLines(lines)
            polydata.GetPointData().SetScalars(scalars)

            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(colors[i])
            actor.GetProperty().SetLineWidth(2)
            renderer.AddActor(actor)

        # Reset camera and ensure renderer is active
        renderer.ResetCamera()
        renderer.GetActiveCamera().Zoom(1.0)  # Ensure camera is not zoomed out
        if renderer.GetActors().GetNumberOfItems() == 0:
            print("No actors added to renderer[2]")
    except Exception as e:
        print(f"Error in display_curves: {e}")

