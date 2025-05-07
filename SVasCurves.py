import vtk

def display_curves(curve_values, colors, renderer, current_step):
    try:
        # Clear all existing actors and reset renderer state
        renderer.RemoveAllViewProps()
        renderer.Clear()

        # Validate indices and plot curves
        for i in range(len(curve_values)):
            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()
            scalars = vtk.vtkFloatArray()
            line = vtk.vtkPolyLine()
            curve_data = curve_values[i]

            # Validate curve data
            if not curve_data or len(curve_data) < 2:
                print(f"Insufficient curve data invalid (<2)")
                continue

            line.GetPointIds().SetNumberOfIds(len(curve_data))
            for j, (x, y) in enumerate(curve_data):
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    print(f"Invalid data point at index {i}, position {j}")
                    continue
                points.InsertNextPoint(x, y, 0)
                line.GetPointIds().SetId(j, j)
                scalars.InsertNextValue(y)
            lines.InsertNextCell(line)

            # Create polydata for curve
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetLines(lines)
            polydata.GetPointData().SetScalars(scalars)

            # Create mapper and actor for curve
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(colors[i])
            actor.GetProperty().SetLineWidth(2)
            renderer.AddActor(actor)

            # Add cube at current_step
            if 0 <= current_step < len(curve_data):
                x, y = curve_data[int(current_step)]
                # Compute x-range from curve_data[:][0]
                x_values = [x_val for x_val, _ in curve_data]
                x_range = max(x_values) - min(x_values) if x_values else 1.0
                cube_length = 0.05 * x_range  # 5% of x-range
                cube_source = vtk.vtkCubeSource()
                cube_source.SetXLength(cube_length)
                cube_source.SetYLength(cube_length)
                cube_source.SetZLength(cube_length)
                cube_mapper = vtk.vtkPolyDataMapper()
                cube_mapper.SetInputConnection(cube_source.GetOutputPort())
                cube_actor = vtk.vtkActor()
                cube_actor.SetMapper(cube_mapper)
                cube_actor.GetProperty().SetColor(colors[i])
                cube_actor.GetProperty().SetOpacity(1.0)
                cube_actor.GetProperty().EdgeVisibilityOn()
                cube_actor.SetPosition(x, y, 1.0)
                renderer.AddActor(cube_actor)

        # Reset camera and adjust clipping range
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        camera.SetClippingRange(0.1, 1000.0)
        camera.Zoom(0.8)
        if renderer.GetActors().GetNumberOfItems() == 0:
            print("No actors added to renderer")
    except Exception as e:
        print(f"Error in display_curves: {e}")

