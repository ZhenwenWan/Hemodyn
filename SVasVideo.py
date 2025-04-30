import numpy as np
import imageio.v2 as imageio
from vtk.util import numpy_support
import vtk

def SVasVideo(ren_win, record_actors, selected_record, interactor, duration):
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.SetInput(ren_win)
    w2i.Update()

    writer = imageio.get_writer("video.mp4", format='ffmpeg', fps=30)

    # Update the selected record actor to show "Recording"
    for actor in record_actors:
        if actor.GetInput() == selected_record:
            actor.SetInput("Recording")
            actor.GetTextProperty().SetColor(1, 0, 0)
        else:
            actor.GetTextProperty().SetColor(1, 1, 1)
    ren_win.Render()

    def capture_frames(obj, event):
        max_frames = duration * 30  # 30 FPS
        if capture_frames.counter < max_frames:
            w2i.Modified()
            w2i.Update()
            vtk_image = w2i.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            arr = numpy_support.vtk_to_numpy(vtk_array).reshape(height, width, -1)
            frame = np.flip(arr, axis=0)
            writer.append_data(frame)
            capture_frames.counter += 1
        else:
            writer.close()
            interactor.RemoveObserver(capture_frames._id)
            for actor in record_actors:
                actor.SetInput(actor.GetInput().replace("Recording", selected_record))
                if actor.GetInput() == selected_record:
                    actor.GetTextProperty().SetColor(0, 1, 0)
                else:
                    actor.GetTextProperty().SetColor(1, 1, 1)
            ren_win.Render()

    capture_frames.counter = 0
    capture_frames._id = interactor.AddObserver("TimerEvent", capture_frames)

