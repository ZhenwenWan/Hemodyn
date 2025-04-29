import vtk
from collections import defaultdict
import importlib
import SVasTimeArr
importlib.reload(SVasTimeArr)
from SVasTimeArr import SVas_TimeArr
import SVasWalkingLegsFunctions
importlib.reload(SVasWalkingLegsFunctions)
from SVasWalkingLegsFunctions import initialize_walking_legs, update_walking_legs
import SVasSliderUtils
importlib.reload(SVasSliderUtils)
from SVasSliderUtils import create_slider_widget
import SVasToggleButton
importlib.reload(SVasToggleButton)
from SVasToggleButton import ToggleButton

# === Load VTP File ===
single_file = "12_AortoFem_Pulse_R_output_verified.vtp"
data_list, polydata, field_time_map = SVas_TimeArr(single_file)

fields = sorted(set(field_time_map.keys()))
time_tags = sorted({k for v in field_time_map.values() for k in v})
num_timesteps = len(time_tags)
walking_num_timesteps = 50  # Walking legs cycle has 50 frames

print(f"Loaded {len(data_list)} frames.")

# === Render Window and 6-Renderer Grid Layout ===
ren_win = vtk.vtkRenderWindow()
ren_win.SetSize(1800, 1200)
renderers = []
for i in range(2):
    for j in range(3):
        ren = vtk.vtkRenderer()
        ren.SetViewport(j / 3.0, 1 - (i + 1) / 2.5, (j + 1) / 3.0, 1 - i / 2.5)
        shade = 0.45 + 0.05 * (i * 3 + j) / 5.0
        ren.SetBackground(shade, shade, shade)
        ren_win.AddRenderer(ren)
        renderers.append(ren)

# === Initialize Walking Legs in Renderer 4 ===
walking_legs_state = initialize_walking_legs(renderers[4])

# === Curve Data Initialization ===
curve_fields = ["pressure", "flow", "wss"]
curve_values = []
for k in range(len(data_list)):
    entry = {f: [] for f in curve_fields}
    for field in curve_fields:
        values = [field_time_map[field][t].GetValue(k) for t in sorted(field_time_map[field])]
        y_min, y_max = min(values), max(values)
        for i, v in enumerate(values):
            y = (v - y_min) / (y_max - y_min) * 100.0
            entry[field].append((i, y))
    curve_values.append(entry)

# === Labels for Renderers ===
def add_label(text, renderer, x, y):
    label = vtk.vtkTextActor()
    label.SetInput(text)
    label.GetTextProperty().SetFontSize(18)
    label.GetTextProperty().SetColor(1, 1, 1)
    label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    label.SetPosition(x, y)
    renderer.AddActor2D(label)

add_label("Pressure", renderers[0], 0.02, 0.9)
add_label("Flow", renderers[1], 0.02, 0.9)
add_label("WSS", renderers[2], 0.02, 0.9)
add_label("Pulse Curves", renderers[3], 0.02, 0.9)
add_label("O2 Dynamics of Walking Legs", renderers[4], 0.02, 0.9)

# === Animation State ===
actors = [vtk.vtkActor() for _ in fields]
step = 0
walking_step = 0  # Separate step for walking legs
is_animating = False

import SVasVideo
importlib.reload(SVasVideo)
from SVasVideo import SVasVideo

# === Video Recording Setup ===
def start_video_recording():
    SVasVideo(ren_win, video_button, interactor)

controller_renderer = vtk.vtkRenderer()
controller_renderer.SetViewport(0.0, 0.0, 1.0, 0.2)
controller_renderer.SetBackground(0.4, 0.4, 0.4)
ren_win.AddRenderer(controller_renderer)

video_button = vtk.vtkTextActor()
video_button.SetInput("Record 10s")
video_button.GetTextProperty().SetFontSize(18)
video_button.GetTextProperty().SetColor(0, 1, 0)
video_button.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
video_button.SetPosition(0.02, 0.85)
controller_renderer.AddActor2D(video_button)

# === Interactor and Shared Camera ===
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(ren_win)
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
shared_camera = vtk.vtkCamera()
for ren in renderers[:3]:
    ren.SetActiveCamera(shared_camera)

# === Camera Toggle ===
def toggle_camera_on(exec):
    if exec:
        for i in range(3):
            ren = renderers[i]
            ren.ResetCamera()
        ren_win.Render()

# === Camera Toggle Button ===
camera_toggle_button = ToggleButton(
    text="Camera",
    x=0.02,
    y=0.02,
    radius=0.05,
    renderer=renderers[0],
    interactor=interactor,
    font_size=24,  # Set a custom font size
    exec_func=toggle_camera_on  # Pass the function to execute
)

# Set individual cameras for curve and walking legs renderers
for i in range(3, 5):
    renderers[i].SetActiveCamera(vtk.vtkCamera())

# === Slider Widgets ===
# Create sliders using the utility function
slider_point, widget_point = create_slider_widget(
    interactor=interactor,
    title="Point",
    min_value=0,
    max_value=len(data_list) - 1,
    x1_pos=0.01,
    x2_pos=0.31,
    y_pos=0.15,
    slider_color=(0.1, 0.8, 0.1)
)

slider_time, widget_timer = create_slider_widget(
    interactor=interactor,
    title="Pulse Frame /80",
    min_value=0,
    max_value=len(data_list) - 1,
    x1_pos=0.35,
    x2_pos=0.65,
    y_pos=0.15,
    slider_color=(0.3, 0.3, 0.8)
)

slider_walking_time, widget_walking_timer = create_slider_widget(
    interactor=interactor,
    title="Walking Frame /50",
    min_value=0,
    max_value=walking_num_timesteps - 1,
    x1_pos=0.68,
    x2_pos=0.98,
    y_pos=0.15,
    slider_color=(0.8, 0.3, 0.3)
)

import SVasCurves
importlib.reload(SVasCurves)
from SVasCurves import display_curves, update_curves

# === Frame Loader ===
def load_frame(idx):
    t, arrays = data_list[idx % len(data_list)]
    area_array = arrays.get("area")
    radii = vtk.vtkDoubleArray()
    radii.SetName("TubeRadius")
    for j in range(area_array.GetNumberOfTuples()):
        r = area_array.GetValue(j) ** 0.5 * 0.1
        radii.InsertNextValue(r)

    field_to_index = {"pressure": 0, "flow": 1, "wss": 2}

    for field in ["pressure", "flow", "wss"]:
        arr = arrays.get(field)
        if arr is None or area_array is None:
            continue

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

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube_filter.GetOutputPort())
        mapper.SetScalarVisibility(True)
        mapper.SetColorModeToMapScalars()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(arr.GetName())
        mapper.SetScalarRange(arr.GetRange())

        index = field_to_index[field]
        actors[index].SetMapper(mapper)
        if actors[index] not in renderers[index].GetActors():
            renderers[index].AddActor(actors[index])

    ipoint = int(slider_point.GetValue())
    if is_animating:
        update_curves(idx, curve_values[ipoint], [renderers[3]])
    else:
        display_curves(curve_values[ipoint], [renderers[3]])

    ren_win.Render()

# === Animation Loop ===
def on_keypress(obj, event):
    global is_animating
    if obj.GetKeySym() == 'a':
        is_animating = not is_animating

def timer_callback(obj, event):
    global step, walking_step
    if is_animating:
        step = (step + 1) % num_timesteps
        walking_step = (walking_step + 1) % walking_num_timesteps
    else:
        step = int(slider_time.GetValue())
        walking_step = int(slider_walking_time.GetValue())
    slider_time.SetValue(step)
    slider_walking_time.SetValue(walking_step)
    load_frame(step)
    update_walking_legs(walking_legs_state, walking_step)
    ren_win.Render()

interactor.AddObserver("KeyPressEvent", on_keypress)
interactor.AddObserver("TimerEvent", timer_callback)
interactor.Initialize()
ren_win.Render()
widget_timer.EnabledOn()
widget_point.EnabledOn()
widget_walking_timer.EnabledOn()
interactor.CreateRepeatingTimer(10)
interactor.Start()

