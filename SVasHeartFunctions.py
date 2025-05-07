import vtk
import glob
import os
import numpy as np
import cv2
from vtk.util.numpy_support import numpy_to_vtk

def load_image_as_vtk(filename, width=512, height=512):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = np.flipud(image)

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(width, height, 1)
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

    vtk_array = numpy_to_vtk(image.ravel(), deep=True)
    vtk_array.SetNumberOfComponents(3)
    image_data.GetPointData().SetScalars(vtk_array)
    image_data.Modified()
    return image_data

def initialize_image_animation(renderer, folder, file_pattern, image_size=(512, 512)):
    """Initialize animation for a series of PNG files in the specified renderer."""
    file_pattern_path = os.path.join(folder, file_pattern)
    files = sorted(glob.glob(file_pattern_path))

    if not files:
        return None, []

    image_data = load_image_as_vtk(files[0], width=image_size[0], height=image_size[1])

    actor = vtk.vtkImageActor()
    actor.SetInputData(image_data)

    renderer.AddActor(actor)
    renderer.GetActiveCamera().SetParallelProjection(True)
    return actor, files

def update_image_animation(index, actor, files, renderer, image_size=(512, 512)):
    """Update animation frame for the specified actor."""
    if not actor or not files:
        return

    file_index = index % len(files)
    image_data = load_image_as_vtk(files[file_index], width=image_size[0], height=image_size[1])

    actor.SetInputData(image_data)
    actor.Modified()
    renderer.Modified()

# Backward compatibility for heart animation
def initialize_heart(renderer):
    """Initialize heart animation in renderer with vtkImageActor."""
    return initialize_image_animation(renderer, "youtube_frames", "frame_0*.png")

def update_heart(index, heart_actor, heart_files, renderer):
    """Update heart animation frame."""
    update_image_animation(index, heart_actor, heart_files, renderer)
