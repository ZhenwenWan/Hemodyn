import vtk
from vtk.util.numpy_support import numpy_to_vtk

def initialize_field_visualization(renderer, fields):
    # Initialize actors for each field
    actors = [vtk.vtkActor() for _ in fields]
    field_to_index = {"pressure": 0, "flow": 1, "wss": 2}

    # Color bar setup
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.667, 0.0)  # Blue (cold) for low, red (warm) for high, matching SVasWalkingLegsFunctions.py
    lut.Build()

    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lut)
    scalar_bar.SetNumberOfLabels(3)
    scalar_bar.SetOrientationToHorizontal()
    scalar_bar.SetPosition(0.01, 0.02)
    scalar_bar.SetWidth(0.96)
    scalar_bar.SetHeight(0.04)
    scalar_bar.Modified()
    renderer.AddActor2D(scalar_bar)

    state = {
        'renderer': renderer,
        'actors': actors,
        'field_to_index': field_to_index,
        'lut': lut,
        'scalar_bar': scalar_bar
    }

    return state

def update_field_visualization(state, data_list, polydata, selected_field, idx):
    renderer = state['renderer']
    actors = state['actors']
    field_to_index = state['field_to_index']
    lut = state['lut']
    scalar_bar = state['scalar_bar']

    t, arrays = data_list[idx % len(data_list)]
    area_array = arrays.get("area")
    radii = vtk.vtkDoubleArray()
    radii.SetName("TubeRadius")
    for j in range(area_array.GetNumberOfTuples()):
        r = area_array.GetValue(j) ** 0.5 * 0.1
        radii.InsertNextValue(r)

    # Clear existing actors except menu actors and scalar bar
    menu_actors = [a for a in renderer.GetActors2D() if isinstance(a, vtk.vtkTextActor) and a.GetInput() in ["Pressure", "Flow", "WSS"]]
    for actor in renderer.GetActors():
        if actor not in menu_actors and actor != scalar_bar:
            renderer.RemoveActor(actor)

    field = selected_field
    arr = arrays.get(field)
    if arr is not None and area_array is not None:
        pdata = vtk.vtkPolyData()
        pdata.SetPoints(polydata.GetPoints())
        pdata.SetLines(polydata.GetLines())
        pdata.GetPointData().AddArray(radii)
        pdata.GetPointData().AddArray(arr)
        pdata.GetPointData().SetActiveScalars(arr.GetName())
        pdata.GetPointData().SetActiveScalars("TubeRadius")

        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputData(pdata)
        tube_filter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
        tube_filter.SetNumberOfSides(12)
        tube_filter.Update()

        # Set scalar range for lookup table based on field data
        scalar_range = arr.GetRange()
        lut.SetTableRange(scalar_range)
        lut.Build()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube_filter.GetOutputPort())
        mapper.SetScalarVisibility(True)
        mapper.SetColorModeToMapScalars()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(arr.GetName())
        mapper.SetScalarRange(scalar_range)
        mapper.SetLookupTable(lut)

        index = field_to_index[field]
        actors[index].SetMapper(mapper)
        renderer.AddActor(actors[index])

    # Ensure scalar bar is added back
    renderer.AddActor2D(scalar_bar)

