import vtk

# Load the .vtu file
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName("case_t0001.vtu")
reader.Update()

# Get the unstructured grid
grid = reader.GetOutput()

# Get the bounds of the grid
bounds = grid.GetBounds()

# Print the bounds
print(f"Coordinate Bounds:")
print(f"X: {bounds[0]} to {bounds[1]}")
print(f"Y: {bounds[2]} to {bounds[3]}")
print(f"Z: {bounds[4]} to {bounds[5]}")
