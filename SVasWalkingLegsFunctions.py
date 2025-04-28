import vtk
import glob
import os
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def create_scalar_bar(lookup_table, title, position, font_size=60, orientation=90):
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lookup_table)
    scalar_bar.SetTitle(title)
    scalar_bar.SetNumberOfLabels(4)
    scalar_bar.SetOrientationToVertical()
    scalar_bar.SetPosition(position[0], position[1])
    scalar_bar.SetWidth(0.1)
    scalar_bar.SetHeight(0.8)
    title_prop = vtk.vtkTextProperty()
    title_prop.SetFontSize(font_size)
    title_prop.SetColor(1, 1, 1)
    title_prop.SetBold(1)
    title_prop.SetOrientation(orientation)
    scalar_bar.SetTitleTextProperty(title_prop)
    scalar_bar.GetLabelTextProperty().SetOrientation(orientation)
    scalar_bar.SetTextPositionToPrecedeScalarBar()
    scalar_bar.Modified()
    return scalar_bar

def create_actor_mapper(poly_data=None, lookup_table=None, scalar_array=None, opacity=1.0):
    mapper = vtk.vtkPolyDataMapper()
    if poly_data:
        mapper.SetInputData(poly_data)
    if lookup_table and scalar_array:
        mapper.SetLookupTable(lookup_table)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(scalar_array)
    else:
        mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    return mapper, actor

def initialize_walking_legs(renderer):
    folder = "../CFD/modelFrame03"
    file_pattern = os.path.join(folder, "case_t*.vtu")
    files = sorted(glob.glob(file_pattern))

    print(f"Loading VTU files from: {os.path.abspath(folder)}")
    print(f"File pattern: {file_pattern}")
    print(f"Found {len(files)} files: {files}")

    if not files:
        raise FileNotFoundError(f"No VTU files found in {os.path.abspath(folder)} matching pattern 'case_t*.vtu'.")

    reader = vtk.vtkXMLUnstructuredGridReader()
    geometry_filter = vtk.vtkGeometryFilter()
    arrow = vtk.vtkArrowSource()

    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetVectorModeToUseVector()
    glyph.SetScaleModeToScaleByScalar()
    glyph.SetColorModeToColorByScalar()
    glyph.OrientOn()
    glyph.SetInputArrayToProcess(1, 0, 0, 0, "velocity_dir")
    glyph.SetInputArrayToProcess(0, 0, 0, 0, "velocity_mag")

    glyph2 = vtk.vtkGlyph3D()
    glyph2.SetSourceConnection(arrow.GetOutputPort())
    glyph2.SetVectorModeToUseVector()
    glyph2.SetScaleModeToScaleByScalar()
    glyph2.SetColorModeToColorByScalar()
    glyph2.OrientOn()
    glyph2.SetInputArrayToProcess(1, 0, 0, 0, "velocity_dir")
    glyph2.SetInputArrayToProcess(0, 0, 0, 0, "velocity_mag")

    lut_velocity = vtk.vtkLookupTable()
    lut_velocity.SetHueRange(0.667, 0.0)
    lut_velocity.Build()

    mapper, actor = create_actor_mapper(None, lut_velocity, "velocity_mag")
    mapper2, actor2 = create_actor_mapper(None, lut_velocity, "velocity_mag")

    lut_o2 = vtk.vtkLookupTable()
    lut_o2.SetHueRange(0.667, 0.0)
    lut_o2.Build()

    o2_mapper, o2_actor = create_actor_mapper(None, lut_o2, "o2vas", opacity=0.5)
    o2_mapper2, o2_actor2 = create_actor_mapper(None, lut_o2, "o2vas", opacity=0.5)

    lut_o2mus = vtk.vtkLookupTable()
    lut_o2mus.SetHueRange(0.333, 0.0)
    lut_o2mus.Build()

    o2mus_mapper, o2mus_actor = create_actor_mapper(None, lut_o2mus, "o2mus", opacity=0.3)
    o2mus_mapper2, o2mus_actor2 = create_actor_mapper(None, lut_o2mus, "o2mus", opacity=0.3)

    proper_poly_data = vtk.vtkPolyData()
    proper_mapper, proper_actor = create_actor_mapper(proper_poly_data, None, None, opacity=1.0)
    proper_actor.GetProperty().SetColor(1, 0, 0)

    renderer.AddActor(actor)
    renderer.AddActor(actor2)
    renderer.AddActor(o2_actor)
    renderer.AddActor(o2_actor2)
    renderer.AddActor(o2mus_actor)
    renderer.AddActor(o2mus_actor2)
    renderer.AddActor(proper_actor)

    velocity_bar = create_scalar_bar(lut_velocity, "Velocity Mag", [0.91, 0.2])
    o2_bar = create_scalar_bar(lut_o2, "O2 Vasculature", [0.01, 0.2])
    o2mus_bar = create_scalar_bar(lut_o2mus, "O2 Muscles", [0.15, 0.2])

    renderer.AddActor2D(velocity_bar)
    renderer.AddActor2D(o2_bar)
    renderer.AddActor2D(o2mus_bar)
    print(f"Velocity bar: Position=[0.91, 0.2], Title Font Size=60, Title Orientation=90, Label Orientation=90")
    print(f"O2 bar: Position=[0.01, 0.2], Title Font Size=60, Title Orientation=90, Label Orientation=90")
    print(f"O2Mus bar: Position=[0.15, 0.2], Title Font Size=60, Title Orientation=90, Label Orientation=90")

    text_actor = vtk.vtkTextActor()
    text_actor.GetTextProperty().SetFontSize(18)
    text_actor.GetTextProperty().SetColor(1, 1, 1)
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    text_actor.SetPosition(0.22, 0.82)
    renderer.AddActor2D(text_actor)

    state = {
        'renderer': renderer,
        'files': files,
        'reader': reader,
        'geometry_filter': geometry_filter,
        'glyph': glyph,
        'glyph2': glyph2,
        'mapper': mapper,
        'mapper2': mapper2,
        'o2_mapper': o2_mapper,
        'o2_mapper2': o2_mapper2,
        'o2mus_mapper': o2mus_mapper,
        'o2mus_mapper2': o2mus_mapper2,
        'actor': actor,
        'actor2': actor2,
        'o2_actor': o2_actor,
        'o2_actor2': o2_actor2,
        'o2mus_actor': o2mus_actor,
        'o2mus_actor2': o2mus_actor2,
        'proper_poly_data': proper_poly_data,
        'proper_mapper': proper_mapper,
        'proper_actor': proper_actor,
        'text_actor': text_actor,
        'current_index': 0,
        'frames_period': 50,
        'frames_phase': 0,
        'a_vec': np.array([0.05, 0.05, 0.05])
    }

    return state

def project_onto_leg_motion(index, coords, velocity_array, phase_offset=0.0, frames_period=50, contract_y=False):
    if contract_y:
        coords = coords.copy()
        coords[:, 1] *= 0.1

    A0 = np.array([-0.045, 0.0, 0.0])
    B0 = np.array([0.045, 0.0, 0.0])
    C0 = np.array([0.146, 0.0, 0.0])

    t = (index % frames_period) / frames_period
    phase = 2 * np.pi * t + phase_offset

    hip_angle = np.deg2rad(20 * np.sin(phase) - 90)
    knee_angle = np.deg2rad(15 + 30 * np.clip(np.sin(phase + np.pi/2), 0, 1)**1.5)

    L1 = np.linalg.norm(B0 - A0)
    L2 = np.linalg.norm(C0 - B0)

    A1 = A0.copy()
    B1 = A1 + L1 * np.array([np.cos(hip_angle), np.sin(hip_angle), 0.0])
    shank_angle = hip_angle - knee_angle
    C1 = B1 + L2 * np.array([np.cos(shank_angle), np.sin(shank_angle), 0.0])

    roU = np.array([
        [np.cos(hip_angle), -np.sin(hip_angle), 0.0],
        [np.sin(hip_angle),  np.cos(hip_angle), 0.0],
        [0.0,        0.0,       1.0]
    ])
    roL = np.array([
        [np.cos(shank_angle), -np.sin(shank_angle), 0.0],
        [np.sin(shank_angle),  np.cos(shank_angle), 0.0],
        [0.0,        0.0,       1.0]
    ])

    new_coords = []
    rotated_velocity = []
    for i, pt in enumerate(coords):
        v = velocity_array[i] if velocity_array is not None else np.zeros(3)
        if pt[0] <= 0.045:
            n_pt = roU @ (pt - A0) + A1
            n_v = roU @ v
        else:
            n_pt = roL @ (pt - B0) + B1
            n_v = roL @ v
        new_coords.append(n_pt)
        rotated_velocity.append(n_v)

    return np.array(new_coords), np.array(rotated_velocity)

def transform_proper_actor(coords, index, frames_period=50):
    A0 = np.array([-0.045, 0.0, 0.0])
    coords = coords.copy()
    coords[:, 1] *= 2.0
    theta = np.deg2rad(90)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,           0.0,          1.0]
    ])
    new_coords = []
    for pt in coords:
        n_pt = rotation @ (pt - A0) + A0
        new_coords.append(n_pt)
    return np.array(new_coords)

def update_walking_legs(state, index):
    files = state['files']
    reader = state['reader']
    renderer = state['renderer']
    geometry_filter = state['geometry_filter']
    glyph = state['glyph']
    glyph2 = state['glyph2']
    mapper = state['mapper']
    mapper2 = state['mapper2']
    o2_mapper = state['o2_mapper']
    o2_mapper2 = state['o2_mapper2']
    o2mus_mapper = state['o2mus_mapper']
    o2mus_mapper2 = state['o2mus_mapper2']
    proper_poly_data = state['proper_poly_data']
    proper_mapper = state['proper_mapper']
    proper_actor = state['proper_actor']
    actor = state['actor']
    actor2 = state['actor2']
    o2_actor = state['o2_actor']
    o2_actor2 = state['o2_actor2']
    o2mus_actor = state['o2mus_actor']
    o2mus_actor2 = state['o2mus_actor2']
    text_actor = state['text_actor']
    frames_period = state['frames_period']

    if not files:
        text_actor.SetInput("Error: No VTU files found in modelFrame03")
        return

    walking_index = index % len(files)
    if walking_index < 0 or walking_index >= len(files):
        text_actor.SetInput(f"Invalid Frame: {walking_index}")
        return
    state['current_index'] = walking_index

    reader.SetFileName(files[walking_index])
    reader.Update()
    geometry_filter.SetInputData(reader.GetOutput())
    geometry_filter.Update()
    poly_data_base = geometry_filter.GetOutput()

    poly_data = vtk.vtkPolyData()
    poly_data.DeepCopy(poly_data_base)
    poly_data2 = vtk.vtkPolyData()
    poly_data2.DeepCopy(poly_data_base)
    poly_data_mus = vtk.vtkPolyData()
    poly_data_mus.DeepCopy(poly_data_base)
    poly_data_mus2 = vtk.vtkPolyData()
    poly_data_mus2.DeepCopy(poly_data_base)

    threshold = vtk.vtkThresholdPoints()
    threshold.SetInputData(poly_data_base)
    threshold.ThresholdByUpper(0.045)
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Points.X")
    threshold.Update()
    proper_poly_data.DeepCopy(threshold.GetOutput())

    points = poly_data.GetPoints()
    coords = vtk_to_numpy(points.GetData())
    velocity_array = poly_data.GetPointData().GetArray("velocity")
    velocity_array_np = vtk_to_numpy(velocity_array) if velocity_array else None

    scaled_coords, vec_np = project_onto_leg_motion(walking_index, coords, velocity_array_np, phase_offset=0.0, frames_period=frames_period, contract_y=True)
    poly_data.GetPoints().SetData(numpy_to_vtk(scaled_coords, deep=True))

    scaled_coords2, vec_np2 = project_onto_leg_motion(walking_index, coords, velocity_array_np, phase_offset=np.pi, frames_period=frames_period, contract_y=True)
    poly_data2.GetPoints().SetData(numpy_to_vtk(scaled_coords2, deep=True))

    scaled_coords_mus, _ = project_onto_leg_motion(walking_index, coords, None, phase_offset=0.0, frames_period=frames_period, contract_y=False)
    poly_data_mus.GetPoints().SetData(numpy_to_vtk(scaled_coords_mus, deep=True))

    scaled_coords_mus2, _ = project_onto_leg_motion(walking_index, coords, None, phase_offset=np.pi, frames_period=frames_period, contract_y=False)
    poly_data_mus2.GetPoints().SetData(numpy_to_vtk(scaled_coords_mus2, deep=True))

    proper_points = proper_poly_data.GetPoints()
    if proper_points and proper_points.GetNumberOfPoints() > 0:
        proper_coords = vtk_to_numpy(proper_points.GetData())
        transformed_coords = transform_proper_actor(proper_coords, walking_index, frames_period)
        proper_poly_data.GetPoints().SetData(numpy_to_vtk(transformed_coords, deep=True))
        print(f"Proper actor: {proper_poly_data.GetNumberOfPoints()} points transformed")
    else:
        print("Proper actor: No points with x <= 0.045")

    if velocity_array_np is not None:
        norms = np.linalg.norm(vec_np, axis=1)
        norms[norms < 1e-9] = 1e-9
        normalized = vec_np / norms[:, None]
        normalized_vtk = numpy_to_vtk(normalized, deep=True)
        normalized_vtk.SetName("velocity_dir")
        poly_data.GetPointData().AddArray(normalized_vtk)
        poly_data.GetPointData().SetActiveVectors("velocity_dir")

        lower, upper = np.percentile(norms, [0, 100])
        clamped = np.clip(norms, lower, upper)
        scaled_log_vtk = numpy_to_vtk(clamped, deep=True)
        scaled_log_vtk.SetName("velocity_mag")
        poly_data.GetPointData().AddArray(scaled_log_vtk)

        bounds = poly_data.GetBounds()
        region_scale = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        mapper.SetScalarRange(clamped.min(), clamped.max())
        glyph.SetScaleFactor(0.015 * region_scale / clamped.max())

    if velocity_array_np is not None:
        norms2 = np.linalg.norm(vec_np2, axis=1)
        norms2[norms2 < 1e-9] = 1e-9
        normalized2 = vec_np2 / norms2[:, None]
        normalized_vtk2 = numpy_to_vtk(normalized2, deep=True)
        normalized_vtk2.SetName("velocity_dir")
        poly_data2.GetPointData().AddArray(normalized_vtk2)
        poly_data2.GetPointData().SetActiveVectors("velocity_dir")

        clamped2 = np.clip(norms2, lower, upper)
        scaled_log_vtk2 = numpy_to_vtk(clamped2, deep=True)
        scaled_log_vtk2.SetName("velocity_mag")
        poly_data2.GetPointData().AddArray(scaled_log_vtk2)

        mapper2.SetScalarRange(clamped2.min(), clamped.max())
        glyph2.SetScaleFactor(0.015 * region_scale / clamped2.max())

    o2_array = poly_data.GetPointData().GetArray("o2vas")
    if o2_array:
        o2_np = vtk_to_numpy(o2_array)
        o2_clamped = np.clip(o2_np, np.percentile(o2_np, 0), np.percentile(o2_np, 100))
        o2_vtk = numpy_to_vtk(o2_clamped, deep=True)
        o2_vtk.SetName("o2vas")
        poly_data.GetPointData().AddArray(o2_vtk)
        o2_mapper.SetInputData(poly_data)
        o2_mapper.SetScalarRange(o2_clamped.min(), o2_clamped.max())
        o2_mapper.Update()
        o2_actor.Modified()

    if o2_array:
        o2_np2 = vtk_to_numpy(o2_array)
        o2_clamped2 = np.clip(o2_np2, np.percentile(o2_np2, 0), np.percentile(o2_np2, 100))
        o2_vtk2 = numpy_to_vtk(o2_clamped2, deep=True)
        o2_vtk2.SetName("o2vas")
        poly_data2.GetPointData().AddArray(o2_vtk2)
        o2_mapper2.SetInputData(poly_data2)
        o2_mapper2.SetScalarRange(o2_clamped2.min(), o2_clamped2.max())
        o2_mapper2.Update()
        o2_actor2.Modified()

    o2mus_array = poly_data_mus.GetPointData().GetArray("o2mus")
    if o2mus_array:
        o2mus_np = vtk_to_numpy(o2mus_array)
        o2mus_clamped = np.clip(o2mus_np, np.percentile(o2mus_np, 0), np.percentile(o2mus_np, 100))
        o2mus_vtk = numpy_to_vtk(o2mus_clamped, deep=True)
        o2mus_vtk.SetName("o2mus")
        poly_data_mus.GetPointData().AddArray(o2mus_vtk)
        o2mus_mapper.SetInputData(poly_data_mus)
        o2mus_mapper.SetScalarRange(o2mus_clamped.min(), o2mus_clamped.max())
        o2mus_mapper.Update()
        o2mus_actor.Modified()

    if o2mus_array:
        o2mus_np2 = vtk_to_numpy(o2mus_array)
        o2mus_clamped2 = np.clip(o2mus_np2, np.percentile(o2mus_np2, 0), np.percentile(o2mus_np2, 100))
        o2mus_vtk2 = numpy_to_vtk(o2mus_clamped2, deep=True)
        o2mus_vtk2.SetName("o2mus")
        poly_data_mus2.GetPointData().AddArray(o2mus_vtk2)
        o2mus_mapper2.SetInputData(poly_data_mus2)
        o2mus_mapper2.SetScalarRange(o2mus_clamped2.min(), o2mus_clamped.max())
        o2mus_mapper2.Update()
        o2mus_actor2.Modified()

    glyph.SetInputData(poly_data)
    glyph.Update()
    mapper.SetInputData(glyph.GetOutput())
    mapper.Update()
    actor.Modified()

    glyph2.SetInputData(poly_data2)
    glyph2.Update()
    mapper2.SetInputData(glyph2.GetOutput())
    mapper2.Update()
    actor2.Modified()

    proper_mapper.SetInputData(proper_poly_data)
    proper_mapper.Update()
    proper_actor.Modified()

    text_actor.SetInput(f"Frame: {walking_index}")
    if index == 0:
        renderer.ResetCamera()

