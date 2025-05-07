import vtk
import glob
import os
import numpy as np
import cv2
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

def ManualPolyData():
    """Create a polygonal outline for visualization."""
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    
    scalex = 0.020
    scaley = 0.030
    scalez = 0.01
    
    points.InsertNextPoint(-2.*scalex+0.005, 0*scaley, 0*scalez)
    points.InsertNextPoint(-1.*scalex+0.005, 1*scaley, 0*scalez)
    points.InsertNextPoint(-1.*scalex+0.005, 4*scaley, 0*scalez)
    points.InsertNextPoint(-2.*scalex+0.005, 5*scaley, 0*scalez)
    points.InsertNextPoint(-3.*scalex+0.005, 5*scaley, 0*scalez)
    points.InsertNextPoint(-4.*scalex+0.005, 4*scaley, 0*scalez)
    points.InsertNextPoint(-4.*scalex+0.005, 1*scaley, 0*scalez)
    points.InsertNextPoint(-3.*scalex+0.005, 0*scaley, 0*scalez)
    
    polygon = vtk.vtkPolygon()
    polygon.GetPointIds().SetNumberOfIds(8)
    for i in range(8):
        polygon.GetPointIds().SetId(i, i)
    cells.InsertNextCell(polygon)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    
    return polydata

def initialize_beatingheart(renderer):
    """Initialize heart animation in the specified renderer."""
    folder = "youtube_frames"
    file_pattern = os.path.join(folder, "frame_0*.png")
    files = sorted(glob.glob(file_pattern))
    print(f"Found {len(files)} heart animation files")

    if not files:
        print("Warning: No PNG files found for heart animation")
        return None, []

    # Load initial PNG image with cv2
    image = cv2.imread(files[0])
    if image is None:
        print(f"Warning: Failed to load PNG {files[0]}")
        return None, []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    image = np.flipud(image)  # Flip vertically to match VTK's coordinate system
    height, width, channels = image.shape
    if channels != 3:
        print(f"Warning: PNG {files[0]} is not RGB (channels={channels})")
        return None, []

    # Create VTK image data for texture
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(width, height, 1)
    image_data.SetSpacing(1, 1, 1)
    image_data.SetOrigin(0, 0, 0)
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
    vtk_array = numpy_to_vtk(image.ravel(), deep=True)
    image_data.GetPointData().SetScalars(vtk_array)

    # Create texture
    texture = vtk.vtkTexture()
    texture.SetInputData(image_data)
    texture.InterpolateOn()

    # Create a plane in normalized viewport coordinates for Renderer 2 (0.666-1.0, 0.6-1.0)
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(0.676, 0.61, 0)  # 2% offset from bottom-left of Renderer 2
    plane.SetPoint1(1.0, 0.61, 0)    # Width: ~0.333 (viewport width)
    plane.SetPoint2(0.676, 0.943, 0) # Height: ~0.333 (viewport height)
    plane.Update()

    # Set texture coordinates
    texture_coords = vtk.vtkFloatArray()
    texture_coords.SetNumberOfComponents(2)
    texture_coords.SetNumberOfTuples(4)
    texture_coords.SetTuple2(0, 0, 0)  # Bottom-left
    texture_coords.SetTuple2(1, 1, 0)  # Bottom-right
    texture_coords.SetTuple2(2, 1, 1)  # Top-right
    texture_coords.SetTuple2(3, 0, 1)  # Top-left
    plane.GetOutput().GetPointData().SetTCoords(texture_coords)

    # Create mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(plane.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetTexture(texture)
    actor.GetProperty().SetOpacity(1.0)

    renderer.AddActor(actor)
    print("Heart actor added to renderer")
    return actor, files

def update_heart(index, heart_actor, heart_files, renderer):
    """Update heart animation with the next PNG frame."""
    if not heart_actor or not heart_files:
        return

    file_index = index % len(heart_files)
    image = cv2.imread(heart_files[file_index])
    if image is None:
        print(f"Warning: Failed to load PNG {heart_files[file_index]}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    image = np.flipud(image)
    height, width, channels = image.shape
    if channels != 3:
        print(f"Warning: PNG {heart_files[file_index]} is not RGB (channels={channels})")
        return

    # Create VTK image data for texture
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(width, height, 1)
    image_data.SetSpacing(1, 1, 1)
    image_data.SetOrigin(0, 0, 0)
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
    vtk_array = numpy_to_vtk(image.ravel(), deep=True)
    image_data.GetPointData().SetScalars(vtk_array)

    # Update texture
    texture = vtk.vtkTexture()
    texture.SetInputData(image_data)
    texture.InterpolateOn()
    heart_actor.SetTexture(texture)
    heart_actor.Modified()
    print(f"Updated heart image with file: {heart_files[file_index]}")

def initialize_walking_legs(renderer, include_heart=False):
    """Initialize walking legs visualization in the specified renderer, optionally with heart animation."""
    folder = "../CFD/modelFrame03"
    file_pattern = os.path.join(folder, "case_t*.vtu")
    files = sorted(glob.glob(file_pattern))

    print(f"Loading VTU files from: {os.path.abspath(folder)}")
    print(f"File pattern: {file_pattern}")
    print(f"Found {len(files)} files")

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

    lut_velocity = vtk.vtkLookupTable()
    lut_velocity.SetHueRange(0.667, 0.0)
    lut_velocity.Build()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetLookupTable(lut_velocity)
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("velocity_mag")

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    glyph2 = vtk.vtkGlyph3D()
    glyph2.SetSourceConnection(arrow.GetOutputPort())
    glyph2.SetVectorModeToUseVector()
    glyph2.SetScaleModeToScaleByScalar()
    glyph2.SetColorModeToColorByScalar()
    glyph2.OrientOn()
    glyph2.SetInputArrayToProcess(1, 0, 0, 0, "velocity_dir")
    glyph2.SetInputArrayToProcess(0, 0, 0, 0, "velocity_mag")

    mapper2 = vtk.vtkPolyDataMapper()
    mapper2.SetLookupTable(lut_velocity)
    mapper2.ScalarVisibilityOn()
    mapper2.SetScalarModeToUsePointFieldData()
    mapper2.SelectColorArray("velocity_mag")

    actor2 = vtk.vtkActor()
    actor2.SetMapper(mapper2)

    lut_o2 = vtk.vtkLookupTable()
    lut_o2.SetHueRange(0.667, 0.0)
    lut_o2.Build()

    o2_mapper = vtk.vtkPolyDataMapper()
    o2_mapper.SetLookupTable(lut_o2)
    o2_mapper.ScalarVisibilityOn()
    o2_mapper.SetScalarModeToUsePointFieldData()
    o2_mapper.SelectColorArray("o2vas")

    o2_actor = vtk.vtkActor()
    o2_actor.SetMapper(o2_mapper)
    o2_actor.GetProperty().SetOpacity(0.5)

    o2_mapper2 = vtk.vtkPolyDataMapper()
    o2_mapper2.SetLookupTable(lut_o2)
    o2_mapper2.ScalarVisibilityOn()
    o2_mapper2.SetScalarModeToUsePointFieldData()
    o2_mapper2.SelectColorArray("o2vas")

    o2_actor2 = vtk.vtkActor()
    o2_actor2.SetMapper(o2_mapper2)
    o2_actor2.GetProperty().SetOpacity(0.5)

    lut_o2mus = vtk.vtkLookupTable()
    lut_o2mus.SetHueRange(0.333, 0.0)
    lut_o2mus.Build()

    o2mus_mapper = vtk.vtkPolyDataMapper()
    o2mus_mapper.SetLookupTable(lut_o2mus)
    o2mus_mapper.ScalarVisibilityOn()
    o2mus_mapper.SetScalarModeToUsePointFieldData()
    o2mus_mapper.SelectColorArray("o2mus")

    o2mus_actor = vtk.vtkActor()
    o2mus_actor.SetMapper(o2mus_mapper)
    o2mus_actor.GetProperty().SetOpacity(0.3)

    o2mus_mapper2 = vtk.vtkPolyDataMapper()
    o2mus_mapper2.SetLookupTable(lut_o2mus)
    o2mus_mapper2.ScalarVisibilityOn()
    o2mus_mapper2.SetScalarModeToUsePointFieldData()
    o2mus_mapper2.SelectColorArray("o2mus")

    o2mus_actor2 = vtk.vtkActor()
    o2mus_actor2.SetMapper(o2mus_mapper2)
    o2mus_actor2.GetProperty().SetOpacity(0.3)

    proper_pdata = ManualPolyData()
    proper_mapper = vtk.vtkPolyDataMapper()
    proper_mapper.SetInputData(proper_pdata)
    proper_actor = vtk.vtkActor()
    proper_actor.SetMapper(proper_mapper)
    proper_actor.GetProperty().SetOpacity(0.3)
    proper_actor.GetProperty().SetColor(1, 0, 0)
    proper_actor.GetProperty().EdgeVisibilityOn()

    renderer.AddActor(actor)
    renderer.AddActor(actor2)
    renderer.AddActor(o2_actor)
    renderer.AddActor(o2_actor2)
    renderer.AddActor(o2mus_actor)
    renderer.AddActor(o2mus_actor2)
    renderer.AddActor(proper_actor)

    o2_bar = vtk.vtkScalarBarActor()
    o2_bar.SetLookupTable(lut_o2)
    o2_bar.SetNumberOfLabels(3)
    o2_bar.SetOrientationToHorizontal()
    o2_bar.SetPosition(0.05, 0.02)
    o2_bar.SetWidth(0.9)
    o2_bar.SetHeight(0.08)
    o2_bar.Modified()
    renderer.AddActor2D(o2_bar)

    text_actor = vtk.vtkTextActor()
    text_actor.GetTextProperty().SetFontSize(10)
    text_actor.GetTextProperty().SetColor(1, 1, 1)
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    text_actor.SetPosition(0.80, 0.90)
    renderer.AddActor2D(text_actor)

    # Initialize heart animation if included
    heart_actor, heart_files = None, []
    if include_heart:
        heart_actor, heart_files = initialize_beatingheart(renderer)

    # Extract O2Vas and O2Mus Curves for Renderer 3 (New 3)
    o2vas_values = []
    o2mus_values = []
    reader.SetFileName(files[0])
    reader.Update()
    poly_data = reader.GetOutput()
    bounds = poly_data.GetBounds()
    x_min, x_max, y_min, y_max = bounds[0], bounds[1], bounds[2], bounds[3]
    percentage = 50.0
    x = x_min + (x_max - x_min) * (percentage / 100.0)
    y = 0.5 * (y_max + y_min)
    z = 0.0
    query_point = [x, y, z]

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(poly_data)
    locator.BuildLocator()
    closest_point_id = locator.FindClosestPoint(query_point)

    for file in files:
        reader.SetFileName(file)
        reader.Update()
        poly_data = reader.GetOutput()
        o2vas_array = poly_data.GetPointData().GetArray("o2vas")
        o2mus_array = poly_data.GetPointData().GetArray("o2mus")
        if o2vas_array:
            o2vas_values.append(o2vas_array.GetValue(closest_point_id))
        else:
            o2vas_values.append(0.0)
        if o2mus_array:
            o2mus_values.append(o2mus_array.GetValue(closest_point_id))
        else:
            o2mus_values.append(0.0)

    if o2vas_values:
        v_min, v_max = min(o2vas_values), max(o2vas_values)
        if v_max != v_min:
            o2vas_values = [(i, (v - v_min) / (v_max - v_min) * 100.0) for i, v in enumerate(o2vas_values)]
        else:
            o2vas_values = [(i, 0.0) for i, v in enumerate(o2vas_values)]
    if o2mus_values:
        m_min, m_max = min(o2mus_values), max(o2mus_values)
        if m_max != m_min:
            o2mus_values = [(i, (v - m_min) / (m_max - m_min) * 100.0) for i, v in enumerate(o2mus_values)]
        else:
            o2mus_values = [(i, 0.0) for i, v in enumerate(o2mus_values)]

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
        'text_actor': text_actor,
        'current_index': 0,
        'frames_period': 50,
        'a_vec': np.array([0.05, 0.05, 0.05]),
        'heart_actor': heart_actor,
        'heart_files': heart_files
    }

    return state, [o2vas_values, o2mus_values]

def project_onto_leg_motion(index, coords, velocity_array, phase_offset=0.0, frames_period=50, contract_y=False):
    """Apply leg motion transformation to coordinates and velocities."""
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

def update_walking_legs(state, index):
    """Update walking legs visualization and heart animation if included."""
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
    actor = state['actor']
    actor2 = state['actor2']
    o2_actor = state['o2_actor']
    o2_actor2 = state['o2_actor2']
    o2mus_actor = state['o2mus_actor']
    o2mus_actor2 = state['o2mus_actor2']
    text_actor = state['text_actor']
    frames_period = state['frames_period']
    heart_actor = state.get('heart_actor')
    heart_files = state.get('heart_files')

    # Update heart animation if included (not used for New 0)
    if heart_actor and heart_files:
        state['heart_index'] = (state.get('heart_index', 0) + 1) % len(heart_files)
        update_heart(state['heart_index'], heart_actor, heart_files, renderer)

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
        o2_min, o2_max = np.percentile(o2_np, [0, 100])
        o2_clamped = np.clip(o2_np, o2_min, o2_max)
        o2_vtk = numpy_to_vtk(o2_clamped, deep=True)
        o2_vtk.SetName("o2vas")
        poly_data.GetPointData().AddArray(o2_vtk)
        o2_mapper.SetInputData(poly_data)
        o2_mapper.SetScalarRange(50, 70)
        o2_mapper.Update()
        o2_actor.Modified()

    if o2_array:
        o2_np2 = vtk_to_numpy(o2_array)
        o2_clamped2 = np.clip(o2_np2, o2_min, o2_max)
        o2_vtk2 = numpy_to_vtk(o2_clamped2, deep=True)
        o2_vtk2.SetName("o2vas")
        poly_data2.GetPointData().AddArray(o2_vtk2)
        o2_mapper2.SetInputData(poly_data2)
        o2_mapper2.SetScalarRange(50, 70)
        o2_mapper2.Update()
        o2_actor2.Modified()

    o2mus_array = poly_data_mus.GetPointData().GetArray("o2mus")
    if o2mus_array:
        o2mus_np = vtk_to_numpy(o2mus_array)
        o2mus_min, o2mus_max = np.percentile(o2mus_np, [0, 100])
        o2mus_clamped = np.clip(o2mus_np, o2mus_min, o2mus_max)
        o2mus_vtk = numpy_to_vtk(o2mus_clamped, deep=True)
        o2mus_vtk.SetName("o2mus")
        poly_data_mus.GetPointData().AddArray(o2mus_vtk)
        o2mus_mapper.SetInputData(poly_data_mus)
        o2mus_mapper.SetScalarRange(50, 70)
        o2mus_mapper.Update()
        o2mus_actor.Modified()

    if o2mus_array:
        o2mus_np2 = vtk_to_numpy(o2mus_array)
        o2mus_min2, o2mus_max2 = np.percentile(o2mus_np2, [0, 100])
        o2mus_clamped2 = np.clip(o2mus_np2, o2mus_min2, o2mus_max2)
        o2mus_vtk2 = numpy_to_vtk(o2mus_clamped2, deep=True)
        o2mus_vtk2.SetName("o2mus")
        poly_data_mus2.GetPointData().AddArray(o2mus_vtk2)
        o2mus_mapper2.SetInputData(poly_data_mus2)
        o2mus_mapper2.SetScalarRange(50, 70)
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
