import vtk
import glob
import os
import numpy as np
import cv2
from vtk.util.numpy_support import numpy_to_vtk

def load_image_as_vtk(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    image = np.flipud(image)

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(512, 512, 1)
    image_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

    vtk_array = numpy_to_vtk(image.ravel(), deep=True)
    vtk_array.SetNumberOfComponents(3)
    image_data.GetPointData().SetScalars(vtk_array)
    image_data.Modified()
    return image_data

def initialize_heart(renderer):
    """Initialize heart animation in Renderer with vtkImageActor."""
    folder = "youtube_frames"
    file_pattern = os.path.join(folder, "frame_0*.png")
    files = sorted(glob.glob(file_pattern))

    if not files:
        return None, []

    image_data = load_image_as_vtk(files[0])

    actor = vtk.vtkImageActor()
    actor.SetInputData(image_data)

    renderer.AddActor(actor)
    renderer.GetActiveCamera().SetParallelProjection(True)
    return actor, files

def update_heart(index, heart_actor, heart_files, renderer):
    """Update heart animation frame."""
    if not heart_actor or not heart_files:
        return

    file_index = index % len(heart_files)
    image_data = load_image_as_vtk(heart_files[file_index])

    heart_actor.SetInputData(image_data)
    heart_actor.Modified()
    renderer.Modified()

