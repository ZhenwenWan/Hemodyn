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
from SVasCurves import display_curves
import SVasMenuUtils
importlib.reload(SVasMenuUtils)
from SVasMenuUtils import create_menu, handle_menu_interaction

# === Configuration ===
VTK_FILE = "12_AortoFem_Pulse_R_output_verified.vtp"
WINDOW_SIZE = (1200, 800)
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

# Get max value for point sliders from first frame's area array
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
    (0.667, 0.2, 1.0, 0.6),  # Renderer 3: O2Vas and O2Mus Curves
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
walking_legs_state, o2_curves = initialize_walking_legs(renderers[0])
o2vas_values, o2mus_values = o2_curves
field_vis_state = initialize_field_visualization(renderers[1], fields)
WALKING_NUM_TIMESTEPS =len(o2vas_values) 

# === Initialize Curve Data for Renderer 2 ===
curve_fields = ["pressure", "flow", "wss", "area", "Re"]
curve_values = []
for k in range(point_slider_max + 1):
    entry = {f: [] for f in curve_fields}
    for field in curve_fields:
        values = [field_time_map[field][t].GetValue(k) for t in range(num_timesteps)]
        y_min, y_max = min(values), max(values)
        for i, v in enumerate(values):
            y = (v - y_min) / (y_max - y_min) * 100.0 if y_max != y_min else 0.0
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
add_label("O2 Curves", renderers[3], 0.02, 0.94)

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
        "options": ["Pressure", "Flow", "WSS", "Area", "Re"],
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
        timer_callback(obj, event)
    ren_win.Render()

interactor.AddObserver("LeftButtonPressEvent", menu_interaction_callback)
interactor.AddObserver("MouseMoveEvent", menu_interaction_callback)

for ren in renderers:
    ren.SetActiveCamera(vtk.vtkCamera())

# === Slider Widgets ===
slider_point, widget_point = create_slider_widget(
    interactor=interactor,
    title="V.Point 1",
    min_value=0,
    max_value=point_slider_max - 1,
    x1_pos=0.68,
    x2_pos=0.98,
    y_pos=0.155,
    slider_color=(1, 0, 0)  # Red
)

slider_point2, widget_point2 = create_slider_widget(
    interactor=interactor,
    title="V.Point 2",
    min_value=0,
    max_value=point_slider_max - 1,
    x1_pos=0.68,
    x2_pos=0.98,
    y_pos=0.105,
    slider_color=(0, 0, 1)  # Blue
)

slider_pulse, widget_timer = create_slider_widget(
    interactor=interactor,
    title="Pulse Frame /80",
    min_value=0,
    max_value=len(data_list) - 1,
    x1_pos=0.35,
    x2_pos=0.65,
    y_pos=0.155,
    slider_color=(0.3, 0.3, 0.8)
)

slider_foot, widget_foot = create_slider_widget(
    interactor=interactor,
    title=f"Walking Frame /{WALKING_NUM_TIMESTEPS}",
    min_value=0,
    max_value=WALKING_NUM_TIMESTEPS - 1,
    x1_pos=0.01,
    x2_pos=0.31,
    y_pos=0.155,
    slider_color=(0.3, 0.8, 0.3)
)

slider_musc, widget_musc = create_slider_widget(
    interactor=interactor,
    title="Muscle Pos %",
    min_value=0,
    max_value=99,
    x1_pos=0.01,
    x2_pos=0.31,
    y_pos=0.105,
    slider_color=(0.3, 0.8, 0.3)
)

# === Frame Loader ===
def load_frame(idx):
    try:
        t, arrays = data_list[idx % len(data_list)]
        ipoint = int(slider_point.GetValue())
        ipoint2 = int(slider_point2.GetValue())
        field = "Re" if selected_field.lower() == "re" else selected_field
        update_field_visualization(field_vis_state, data_list, polydata, field, idx, ipoint, ipoint2)
        _curve_values = [curve_values[ipoint][field], curve_values[ipoint2][field]]
        pulse_step = int(slider_pulse.GetValue())
        walking_step = int(slider_foot.GetValue())
        display_curves(_curve_values, [(1,0,0),(0,0,1)], renderers[2], pulse_step)
        display_curves([o2vas_values, o2mus_values], [(1,0,0),(0,0,1)], renderers[3], walking_step)
        renderers[2].ResetCamera()
        renderers[3].ResetCamera()
        ren_win.Render()
    except Exception as e:
        print(f"Error in load_frame: {e}")

# === Animation Control ===
def on_keypress(obj, event):
    global is_animating
    if obj.GetKeySym() == 'a':
        is_animating = not is_animating

def timer_callback(obj, event):
    global step, walking_step
    timer_ms, frames_per_tick = FRAME_RATES[selected_mode]
    if is_animating:
        step = (step + frames_per_tick) % num_timesteps
        walking_step = (walking_step + frames_per_tick) % WALKING_NUM_TIMESTEPS
        slider_pulse.SetValue(step)
        slider_foot.SetValue(walking_step)
    else:
        step = int(slider_pulse.GetValue())
        walking_step = int(slider_foot.GetValue())
    try:
        load_frame(step)
        update_walking_legs(walking_legs_state, walking_step)
    except Exception as e:
        print(f"Error in timer_callback: {e}")
    ren_win.Render()

# === Initialize and Start ===
for renderer in renderers:
    renderer.ResetCamera()
interactor.AddObserver("KeyPressEvent", on_keypress)
interactor.AddObserver("TimerEvent", timer_callback)
interactor.Initialize()
ren_win.Render()
interactor.CreateRepeatingTimer(FRAME_RATES[selected_mode][0])
interactor.Start()

