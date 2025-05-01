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
import SVasFieldVisualization
importlib.reload(SVasFieldVisualization)
from SVasFieldVisualization import initialize_field_visualization, update_field_visualization
import SVasCurves
importlib.reload(SVasCurves)
from SVasCurves import display_curves, update_curves
import SVasMenuUtils
importlib.reload(SVasMenuUtils)
from SVasMenuUtils import create_menu, handle_menu_interaction

# === Configuration ===
VTK_FILE = "12_AortoFem_Pulse_R_output_verified.vtp"
WINDOW_SIZE = (1200, 800)
WALKING_NUM_TIMESTEPS = 50
FRAME_RATES = {
    "Roaming": (100, 1),
    "Walking": (50, 1),
    "Running": (25, 2)
}

# === Load VTP Data ===
data_list, polydata, field_time_map = SVas_TimeArr(VTK_FILE)
fields = sorted(set(field_time_map.keys()))
time_tags = sorted({k for v in field_time_map.values() for k in v})
num_timesteps = len(time_tags)
print(f"Loaded {len(data_list)} frames.")

# Get max value for slider_point from first frame's area array
_, arrays = data_list[0]
point_slider_max = arrays.get("area").GetNumberOfTuples() - 1

# === Initialize Render Window and Renderers ===
ren_win = vtk.vtkRenderWindow()
ren_win.SetSize(*WINDOW_SIZE)

renderers = []
viewport_configs = [
    (0.0, 0.2, 0.333, 1.0),  # Renderer 0: Walking Legs
    (0.333, 0.2, 0.667, 1.0),  # Renderer 1: Selected Field
    (0.667, 0.6, 1.0, 1.0),  # Renderer 2: Pulse Curves
    (0.667, 0.2, 1.0, 0.6),  # Renderer 3: Empty
    (0.0, 0.0, 1.0, 0.2)  # Renderer 4: Controller
]

for idx, viewport in enumerate(viewport_configs):
    ren = vtk.vtkRenderer()
    ren.SetViewport(viewport)
    shade = 0.45 + 0.05 * idx / 5.0
    ren.SetBackground(shade, shade, shade)
    ren_win.AddRenderer(ren)
    renderers.append(ren)

renderers[4].SetBackground(0.4, 0.4, 0.4)

# === Initialize Visualizations ===
walking_legs_state = initialize_walking_legs(renderers[0])
field_vis_state = initialize_field_visualization(renderers[1], fields)

# === Initialize Curve Data ===
curve_fields = ["pressure", "flow", "wss"]
curve_values = []
for k in range(point_slider_max + 1):
    entry = {f: [] for f in curve_fields}
    for field in curve_fields:
        values = [field_time_map[field][t].GetValue(k) for t in sorted(field_time_map[field])]
        y_min, y_max = min(values), max(values)
        for i, v in enumerate(values):
            y = (v - y_min) / (y_max - y_min) * 100.0
            entry[field].append((i, y))
    curve_values.append(entry)

# === Add Labels ===
def add_label(text, renderer, x, y):
    label = vtk.vtkTextActor()
    label.SetInput(text)
    label.GetTextProperty().SetFontSize(18)
    label.GetTextProperty().SetColor(1, 1, 1)
    label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    label.SetPosition(x, y)
    renderer.AddActor2D(label)
    return label

add_label("Pulse Curves", renderers[2], 0.02, 0.94)

# === Setup Menus ===
menu_configs = [
    {
        "renderer": renderers[0],
        "options": ["Roaming", "Walking", "Running"],
        "default": "Walking",
        "position": (0.02, 0.94, 0.04)
    },
    {
        "renderer": renderers[1],
        "options": ["Pressure", "Flow", "WSS"],
        "default": "pressure",
        "position": (0.02, 0.94, 0.04)
    },
    {
        "renderer": renderers[4],
        "options": ["Record 10s", "Record 30s"],
        "default": None,
        "position": (0.02, 0.18, 0.12)
    }
]

menu_actors_dict = {}
selected_options = {}
for config in menu_configs:
    actors, selected = create_menu(
        config["renderer"],
        config["options"],
        config["default"],
        config["position"]
    )
    menu_actors_dict[config["renderer"]] = actors
    selected_options[config["renderer"]] = selected

selected_mode = selected_options[renderers[0]]
selected_field = selected_options[renderers[1]]
selected_record = selected_options[renderers[4]]

# === Animation State ===
step = 0
walking_step = 0
is_animating = False

# === Video Recording ===
def start_video_recording():
    duration = 30 if selected_record == "Record 30s" else 10
    SVasVideo(ren_win, menu_actors_dict[renderers[4]], selected_record, interactor, duration)

# === Interactor Setup ===
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(ren_win)
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

def menu_interaction_callback(obj, event):
    global selected_mode, selected_field, selected_record
    selected_options[renderers[0]] = selected_mode
    selected_options[renderers[1]] = selected_field
    selected_options[renderers[4]] = selected_record
    new_selections = handle_menu_interaction(
        obj, event, ren_win, interactor, menu_configs, menu_actors_dict, selected_options
    )
    selected_mode = new_selections[renderers[0]]
    selected_field = new_selections[renderers[1]]
    selected_record = new_selections[renderers[4]]
    if event == "LeftButtonPressEvent" and selected_record:
        start_video_recording()
    if event == "LeftButtonPressEvent" and (0.333 <= obj.GetEventPosition()[0] / ren_win.GetSize()[0] <= 0.667):
        load_frame(step)
    ren_win.Render()

interactor.AddObserver("LeftButtonPressEvent", menu_interaction_callback)
interactor.AddObserver("MouseMoveEvent", menu_interaction_callback)

for ren in renderers:
    ren.SetActiveCamera(vtk.vtkCamera())

# === Slider Widgets ===
slider_point, widget_point = create_slider_widget(
    interactor=interactor,
    title="Point",
    min_value=0,
    max_value=point_slider_max,
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
    max_value=WALKING_NUM_TIMESTEPS - 1,
    x1_pos=0.01,
    x2_pos=0.31,
    y_pos=0.125,
    slider_color=(0.8, 0.3, 0.3)
)

# === Frame Loader ===
def load_frame(idx):
    t, arrays = data_list[idx % len(data_list)]
    update_field_visualization(field_vis_state, data_list, polydata, selected_field, idx, int(slider_point.GetValue()))
    renderers[3].RemoveAllViewProps()
    ipoint = int(slider_point.GetValue())
    if is_animating:
        update_curves(idx, curve_values[ipoint], [renderers[2]])
    else:
        display_curves(curve_values[ipoint], [renderers[2]])
    ren_win.Render()

# === Animation Control ===
def on_keypress(obj, event):
    global is_animating
    if obj.GetKeySym() == 'a':
        is_animating = not is_animating
    load_frame(step)
    ren_win.Render()

def timer_callback(obj, event):
    global step, walking_step
    timer_ms, frames_per_tick = FRAME_RATES[selected_mode]
    if is_animating:
        step = (step + frames_per_tick) % num_timesteps
        walking_step = (walking_step + frames_per_tick) % WALKING_NUM_TIMESTEPS
    else:
        step = int(slider_time.GetValue())
        walking_step = int(slider_walking_time.GetValue())
    slider_time.SetValue(step)
    slider_walking_time.SetValue(walking_step)
    load_frame(step)
    update_walking_legs(walking_legs_state, walking_step)
    ren_win.Render()
    interactor.DestroyTimer()
    interactor.CreateRepeatingTimer(timer_ms)

# === Initialize and Start ===
for renderer in renderers:
    renderer.ResetCamera()

interactor.AddObserver("KeyPressEvent", on_keypress)
interactor.AddObserver("TimerEvent", timer_callback)
interactor.Initialize()
ren_win.Render()

widget_timer.EnabledOn()
widget_point.EnabledOn()
widget_walking_timer.EnabledOn()
interactor.CreateRepeatingTimer(FRAME_RATES[selected_mode][0])
interactor.Start()

