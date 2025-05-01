import vtk

def create_menu(renderer, options, default_selection, position):
    x, y_start, y_step = position
    actors = []
    selected = default_selection

    for i, option in enumerate(options):
        actor = vtk.vtkTextActor()
        actor.SetInput(option)
        actor.GetTextProperty().SetFontSize(18)
        actor.GetTextProperty().SetColor(1, 1, 1)
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.SetPosition(x, y_start - i * y_step)
        renderer.AddActor2D(actor)
        actors.append(actor)
        if option.lower() == default_selection:
            actor.GetTextProperty().SetColor(0, 1, 0)

    return actors, selected

def handle_menu_interaction(obj, event, ren_win, interactor, menu_configs, menu_actors_dict, selected_options):
    x, y = interactor.GetEventPosition()
    width, height = ren_win.GetSize()
    nx = x / width
    ny = y / height
    new_selections = selected_options.copy()

    # Renderer 0: Roaming/Walking/Running
    config_0 = menu_configs[0]
    renderer_0 = config_0["renderer"]
    actors_0 = menu_actors_dict[renderer_0]
    if 0.0 <= nx <= 0.333 and 0.2 <= ny <= 1.0:
        ny_remapped = (ny - 0.2) / 0.8
        hovered = None
        for i, actor in enumerate(actors_0):
            y_pos = 0.94 - i * 0.04
            if abs(ny_remapped - y_pos) < 0.02:
                hovered = actor.GetInput()
                if event == "LeftButtonPressEvent":
                    new_selections[renderer_0] = hovered
            if actor.GetInput() == new_selections[renderer_0]:
                actor.GetTextProperty().SetColor(0, 1, 0)
            elif actor.GetInput() == hovered:
                actor.GetTextProperty().SetColor(0, 0, 1)
            else:
                actor.GetTextProperty().SetColor(1, 1, 1)
    else:
        for actor in actors_0:
            if actor.GetInput() == new_selections[renderer_0]:
                actor.GetTextProperty().SetColor(0, 1, 0)
            else:
                actor.GetTextProperty().SetColor(1, 1, 1)

    # Renderer 1: Pressure/Flow/WSS
    config_1 = menu_configs[1]
    renderer_1 = config_1["renderer"]
    actors_1 = menu_actors_dict[renderer_1]
    if 0.333 <= nx <= 0.667 and 0.2 <= ny <= 1.0:
        ny_remapped = (ny - 0.2) / 0.8
        hovered = None
        for i, actor in enumerate(actors_1):
            y_pos = 0.94 - i * 0.04
            if abs(ny_remapped - y_pos) < 0.02:
                hovered = actor.GetInput()
                if event == "LeftButtonPressEvent":
                    new_selections[renderer_1] = hovered.lower()
            if actor.GetInput().lower() == new_selections[renderer_1]:
                actor.GetTextProperty().SetColor(0, 1, 0)
            elif actor.GetInput() == hovered:
                actor.GetTextProperty().SetColor(0, 0, 1)
            else:
                actor.GetTextProperty().SetColor(1, 1, 1)
    else:
        for actor in actors_1:
            if actor.GetInput().lower() == new_selections[renderer_1]:
                actor.GetTextProperty().SetColor(0, 1, 0)
            else:
                actor.GetTextProperty().SetColor(1, 1, 1)

    # Renderer 4: Record 10s/30s
    config_4 = menu_configs[2]
    renderer_4 = config_4["renderer"]
    actors_4 = menu_actors_dict[renderer_4]
    if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 0.2:
        ny_remapped = ny / 0.2
        hovered = None
        for i, actor in enumerate(actors_4):
            y_pos = 0.18 - i * 0.12
            if abs(ny_remapped - y_pos) < 0.02:
                hovered = actor.GetInput()
                if event == "LeftButtonPressEvent":
                    new_selections[renderer_4] = hovered
            if actor.GetInput() == new_selections[renderer_4]:
                actor.GetTextProperty().SetColor(0, 1, 0)
            elif actor.GetInput() == hovered:
                actor.GetTextProperty().SetColor(0, 0, 1)
            else:
                actor.GetTextProperty().SetColor(1, 1, 1)
    else:
        for actor in actors_4:
            if actor.GetInput() == new_selections[renderer_4]:
                actor.GetTextProperty().SetColor(0, 1, 0)
            else:
                actor.GetTextProperty().SetColor(1, 1, 1)

    return new_selections

