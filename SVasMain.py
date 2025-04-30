import vtk
from collections import defaultdict
import importlib
import SVasTimeArr
importlib.reload(SVasTimeArr)
from SVasTimeArr import SVas_TimeArr
import SVasVideo
importlib.reload(SVasVideo)
from SVasVideo import SVasVideo
import SVasWalkingLegsFunctions
importlib.reload(SVasWalkingLegsFunctions)
from SVasWalkingLegsFunctions import initialize_walking_legs, update_walking_legs
import SVasSliderUtils
importlib.reload(SVasSliderUtils)
from SVasSliderUtils import create_slider_widget

# === Load VTP File ===
single_file = "12_AortoFem_Pulse_R_output_verified.vtp"
data_list, polydata, field_time_map = SVas_TimeArr(single_file)

fields = sorted(set(field_time_map.keys()))
time_tags = sorted({k for v in field_time_map.values() for k in v})
num_timesteps = len(time_tags)
walking_num_timesteps = 50

print(f"Loaded {len(data_list)} frames.")

# === Render Window and 5-Renderer Layout ===
ren_win = vtk.vtkRenderWindow()
ren_win.SetSize(1200, 800)
renderers = []
for i in range(2):
    for j in range(3):
        if i == 1 and j == 2:
            continue
        ren = vtk.vtkRenderer()
        if i == 0 and j == 0:  # renderers[0]: Walking Legs
            ren.SetViewport(0.0, 0.2, 0.333, 1.0)
        elif i == 0 and j == 1:  # renderers[1]: Selected Field
            ren.SetViewport(0.333, 0.2, 0.667, 1.0)
        elif i == 0 and j == 2:  # renderers[2]: Pulse Curves
            ren.SetViewport(0.667, 0.6, 1.0, 1.0)
        elif i == 1 and j == 0:  # renderers[3]: Empty
            ren.SetViewport(0.667, 0.2, 1.0, 0.6)
        elif i == 1 and j == 1:  # renderers[4]: Controller
            ren.SetViewport(0.0, 0.0, 1.0, 0.2)
        shade = 0.45 + 0.05 * (i * 3 + j) / 5.0
        ren.SetBackground(shade, shade, shade)
        ren_win.AddRenderer(ren)
        renderers.append(ren)

renderers[4].SetBackground(0.4, 0.4, 0.4)

# === Initialize Walking Legs in Renderer 0 ===
walking_legs_state = initialize_walking_legs(renderers[0])

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
    return label

selected_field = "pressure"
field_label = add_label(f"Selected: {selected_field.capitalize()}", renderers[1], 0.02, 0.94)
add_label("Pulse Curves", renderers[2], 0.02, 0.94)

# === Menu for Roaming/Walking/Running ===
menu_options = ["Roaming", "Walking", "Running"]
menu_actors = []
selected_mode = "Walking"  # Default mode
frame_rates = {"Roaming": (100, 1), "Walking": (50, 1), "Running": (25, 2)}  # (timer_ms, frames_per_tick)

for i, option in enumerate(menu_options):
    actor = vtk.vtkTextActor()
    actor.SetInput(option)
    actor.GetTextProperty().SetFontSize(18)
    actor.GetTextProperty().SetColor(1, 1, 1)  # White by default
    actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    actor.SetPosition(0.02, 0.94 - i * 0.04)
    renderers[0].AddActor2D(actor)
    menu_actors.append(actor)

# Set initial selected mode color
for actor in menu_actors:
    if actor.GetInput() == selected_mode:
        actor.GetTextProperty().SetColor(0, 1, 0)  # Green for selected

# === Animation State ===
actors = [vtk.vtkActor() for _ in fields]
step = 0
walking_step = 0
is_animating = False

# === Video Recording Setup ===
def start_video_recording():
    SVasVideo(ren_win, video_button, interactor)

video_button = vtk.vtkTextActor()
video_button.SetInput("Record 10s")
video_button.GetTextProperty().SetFontSize(24)
video_button.GetTextProperty().SetColor(0, 1, 0)
video_button.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
video_button.SetPosition(0.02, 0.05)
renderers[4].AddActor2D(video_button)

def video_button_callback(obj, event):
    if event == "LeftButtonPressEvent":
        x, y = interactor.GetEventPosition()
        width, height = ren_win.GetSize()
        nx = x / width
        ny = y / height
        if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 0.2:
            if abs(nx - 0.02) < 0.1 and abs(ny - 0.05) < 0.05:
                start_video_recording()

# === Menu Interaction ===
def menu_interaction_callback(obj, event):
    global selected_mode
    x, y = interactor.GetEventPosition()
    width, height = ren_win.GetSize()
    nx = x / width
    ny = y / height
    # Check if in renderers[0] viewport (0.0, 0.2, 0.333, 1.0)
    if 0.0 <= nx <= 0.333 and 0.2 <= ny <= 1.0:
        # Map ny to renderer[0]'s normalized coordinates (0.2 to 1.0 -> 0.0 to 1.0)
        ny_remapped = (ny - 0.2) / 0.8
        hovered = None
        for i, actor in enumerate(menu_actors):
            y_pos = 0.94 - i * 0.04
            if abs(ny_remapped - y_pos) < 0.02:
                hovered = actor.GetInput()
                if event == "LeftButtonPressEvent":
                    selected_mode = hovered
            # Update colors
            if actor.GetInput() == selected_mode:
                actor.GetTextProperty().SetColor(0, 1, 0)  # Green for selected
            elif actor.GetInput() == hovered:
                actor.GetTextProperty().SetColor(0, 0, 1)  # Blue for hover
            else:
                actor.GetTextProperty().SetColor(1, 1, 1)  # White otherwise
    else:
        # Reset to white except for selected
        for actor in menu_actors:
            if actor.GetInput() == selected_mode:
                actor.GetTextProperty().SetColor(0, 1, 0)
            else:
                actor.GetTextProperty().SetColor(1, 1, 1)
    ren_win.Render()

# === Interactor and Individual Cameras ===
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(ren_win)
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
interactor.AddObserver("LeftButtonPressEvent", video_button_callback)
interactor.AddObserver("LeftButtonPressEvent", menu_interaction_callback)
interactor.AddObserver("MouseMoveEvent", menu_interaction_callback)
for ren in renderers:
    ren.SetActiveCamera(vtk.vtkCamera())

# === Slider Widgets ===
slider_point, widget_point = create_slider_widget(
    interactor=interactor,
    title="Point",
    min_value=0,
    max_value=len(data_list) - 1,
    x1_pos=0.68,
    x2_pos=0.98,
    y_pos=0.125,
    slider_color=(0.1, 0.8, 0.1)
)

slider_time, widget_timer = create_slider_widget(
    interactor=interactor,
    title="Pulse Frame /80",
    min_value=0,
    max_value=len(data_list) - 1,
    x1_pos=0.35,
    x2_pos=0.65,
    y_pos=0.125,
    slider_color=(0.3, 0.3, 0.8)
)

slider_walking_time, widget_walking_timer = create_slider_widget(
    interactor=interactor,
    title="Walking Frame /50",
    min_value=0,
    max_value=walking_num_timesteps - 1,
    x1_pos=0.01,
    x2_pos=0.31,
    y_pos=0.125,
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

    for actor in renderers[1].GetActors():
        if actor != field_label:
            renderers[1].RemoveActor(actor)

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

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube_filter.GetOutputPort())
        mapper.SetScalarVisibility(True)
        mapper.SetColorModeToMapScalars()
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(arr.GetName())
        mapper.SetScalarRange(arr.GetRange())

        index = field_to_index[field]
        actors[index].SetMapper(mapper)
        renderers[1].AddActor(actors[index])

    renderers[3].RemoveAllViewProps()

    ipoint = int(slider_point.GetValue())
    if is_animating:
        update_curves(idx, curve_values[ipoint], [renderers[2]])
    else:
        display_curves(curve_values[ipoint], [renderers[2]])

    ren_win.Render()

# === Field Selection and Animation Loop ===
def on_keypress(obj, event):
    global is_animating, selected_field
    key = obj.GetKeySym()
    if key == 'a':
        is_animating = not is_animating
    elif key == 'p':
        selected_field = "pressure"
        field_label.SetInput("Selected: Pressure")
    elif key == 'f':
        selected_field = "flow"
        field_label.SetInput("Selected: Flow")
    elif key == 'w':
        selected_field = "wss"
        field_label.SetInput("Selected: WSS")
    load_frame(step)
    ren_win.Render()

def timer_callback(obj, event):
    global step, walking_step
    timer_ms, frames_per_tick = frame_rates[selected_mode]
    if is_animating:
        step = (step + frames_per_tick) % num_timesteps
        walking_step = (walking_step + frames_per_tick) % walking_num_timesteps
    else:
        step = int(slider_time.GetValue())
        walking_step = int(slider_walking_time.GetValue())
    slider_time.SetValue(step)
    slider_walking_time.SetValue(walking_step)
    load_frame(step)
    update_walking_legs(walking_legs_state, walking_step)
    ren_win.Render()
    # Update timer interval
    interactor.DestroyTimer()
    interactor.CreateRepeatingTimer(timer_ms)

for idx in range(len(renderers)):
    renderers[idx].ResetCamera()
interactor.AddObserver("KeyPressEvent", on_keypress)
interactor.AddObserver("TimerEvent", timer_callback)
interactor.Initialize()
ren_win.Render()
widget_timer.EnabledOn()
widget_point.EnabledOn()
widget_walking_timer.EnabledOn()
interactor.CreateRepeatingTimer(frame_rates[selected_mode][0])
interactor.Start()

