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
import SVasHeartFunctions
importlib.reload(SVasHeartFunctions)
from SVasHeartFunctions import initialize_heart, update_heart
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
    (0.0, 0.6, 0.333, 1.0),     # New 0: Walking legs (Current 0, no heart)
    (0.333, 0.6, 0.666, 1.0),   # New 1: Current 1 (field visualization)
    (0.666, 0.6, 1.0, 1.0),     # New 2: Heart animation (from Current 0)
    (0.0, 0.2, 0.333, 0.6),     # New 3: Current 3 (o2vas, o2mus curves)
    (0.333, 0.2, 0.666, 0.6),   # New 4: Current 2 (pulse curves)
    (0.666, 0.2, 1.0, 0.6),     # New 5: Empty
    (0.0, 0.0, 1.0, 0.2)        # New 6: Current 4 (controller)
]

for idx, viewport in enumerate(viewport_configs):
    ren = vtk.vtkRenderer()
    ren.SetViewport(viewport)
    shade = 0.45 + 0.05 * idx / 7.0  # Gradient for distinction
    ren.SetBackground(shade, shade, shade)
    ren_win.AddRenderer(ren)
    renderers.append(ren)

renderers[6].SetBackground(0.4, 0.4, 0.4)  # Match original controller background
renderers[2].SetBackground(0.3, 0.3, 0.3)  # Ensure neutral background for heart colors

# === Initialize Visualizations ===
# New 0: Walking legs from Current 0 (no heart animation)
walking_legs_state, o2_curves = initialize_walking_legs(renderers[0])
o2vas_values, o2mus_values = o2_curves

# New 1: Current 1 (field visualization)
field_vis_state = initialize_field_visualization(renderers[1], fields)

# New 2: Heart animation from Current 0
heart_actor, heart_files = initialize_heart(renderers[2])

# New 3: Current 3 (o2vas, o2mus curves)
WALKING_NUM_TIMESTEPS = len(o2vas_values)
display_curves([o2vas_values, o2mus_values], [(1, 0, 0), (0, 0, 1)], renderers[3], 0)

# New 4: Current 2 (pulse curves)
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

# New 5: Empty (no content)

# New 6: Current 4 (controller: sliders, menus)
# (Sliders and menus initialized below)

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

add_label("O2 Curves", renderers[3], 0.02, 0.94)
add_label("Pulse Curves", renderers[4], 0.02, 0.94)

# === Setup Menus ===
menu_configs = [
    {
        "renderer": renderers[0],
        "options": ["Roaming", "Walking", "Running"],
        "default": "Walking",
        "position": (0.02, 0.94, 0.04),
        "font_size": 12  # Smaller font size for Renderer 0
    },
    {
        "renderer": renderers[1],
        "options": ["Pressure", "Flow", "WSS", "Area", "Re"],
        "default": "pressure",
        "position": (0.02, 0.94, 0.04),
        "font_size": 12  # Smaller font size for Renderer 1
    },
    {
        "renderer": renderers[6],
        "options": ["Record 10s", "Record 30s"],
        "default": None,
        "position": (0.02, 0.18, 0.12),
        "font_size": 18  # Original font size for Renderer 6
    }
]

menu_actors_dict = {}
selected_options = {}
for config in menu_configs:
    actors, selected = create_menu(
        config["renderer"],
        config["options"],
        config["default"],
        config["position"],
        font_size=config.get("font_size", 18)
    )
    menu_actors_dict[config["renderer"]] = actors
    selected_options[config["renderer"]] = selected

selected_mode = selected_options[renderers[0]]
selected_field = selected_options[renderers[1]]
selected_record = selected_options[renderers[6]]

# === Animation State ===
step = 0
walking_step = 0
is_animating = False

# === Video Recording ===
def start_video_recording():
    duration = 30 if selected_record == "Record 30s" else 10
    SVasVideo(ren_win, menu_actors_dict[renderers[6]], selected_record, interactor, duration)

# === Interactor Setup ===
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(ren_win)
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

def menu_interaction_callback(obj, event):
    global selected_mode, selected_field, selected_record
    selected_options[renderers[0]] = selected_mode
    selected_options[renderers[1]] = selected_field
    selected_options[renderers[6]] = selected_record
    new_selections = handle_menu_interaction(
        obj, event, ren_win, interactor, menu_configs, menu_actors_dict, selected_options
    )
    selected_mode = new_selections[renderers[0]]
    selected_field = new_selections[renderers[1]]
    selected_record = new_selections[renderers[6]]
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
    slider_color=(1, 0, 0)
)

slider_point2, widget_point2 = create_slider_widget(
    interactor=interactor,
    title="V.Point 2",
    min_value=0,
    max_value=point_slider_max - 1,
    x1_pos=0.68,
    x2_pos=0.98,
    y_pos=0.105,
    slider_color=(0, 0, 1)
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
        display_curves(_curve_values, [(1, 0, 0), (0, 0, 1)], renderers[4], pulse_step)  # New 4 (Current 2)
        display_curves([o2vas_values, o2mus_values], [(1, 0, 0), (0, 0, 1)], renderers[3], walking_step)  # New 3 (Current 3)
        renderers[4].ResetCamera()
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
        update_heart(walking_step, heart_actor, heart_files, renderers[2])  # Update heart in New 2
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
