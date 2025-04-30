import vtk

def create_slider_widget(interactor, title, min_value, max_value, x1_pos, x2_pos, y_pos, slider_color):
    """
    Create a VTK slider representation and widget.
    
    Args:
        interactor: vtkRenderWindowInteractor for the widget.
        title (str): Slider title text.
        min_value (float): Minimum slider value.
        max_value (float): Maximum slider value.
        y_position (float): Normalized Y-coordinate for slider position.
        slider_color (tuple): RGB color for the slider (e.g., (0.3, 0.3, 0.8)).
    
    Returns:
        tuple: (vtkSliderRepresentation2D, vtkSliderWidget)
    """
    slider = vtk.vtkSliderRepresentation2D()
    slider.SetMinimumValue(min_value)
    slider.SetMaximumValue(max_value)
    slider.SetValue(min_value)
    slider.SetTitleText(title)
    slider.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider.GetPoint1Coordinate().SetValue(x1_pos, y_pos)
    slider.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider.GetPoint2Coordinate().SetValue(x2_pos, y_pos)
    slider.SetSliderLength(0.03)
    slider.SetSliderWidth(0.05)
    slider.SetEndCapLength(0.02)
    slider.SetTubeWidth(0.008)
    slider.SetLabelFormat("%0.0f")
    slider.SetTitleHeight(0.1)
    slider.SetLabelHeight(0.1)
    slider.GetSliderProperty().SetColor(*slider_color)
    slider.GetSelectedProperty().SetColor(0.6, 0.1, 0.1)
    slider.GetTubeProperty().SetColor(0.2, 0.2, 0.2)

    widget = vtk.vtkSliderWidget()
    widget.SetInteractor(interactor)
    widget.SetRepresentation(slider)
    widget.SetAnimationModeToJump()
    widget.EnabledOn()

    return slider, widget

