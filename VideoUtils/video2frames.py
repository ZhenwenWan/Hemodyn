import ffmpeg
import os

# Create output directory
os.makedirs("youtube_frames", exist_ok=True)

# Extract frames
ffmpeg.input('Heart_youtube.mp4').output('youtube_frames/frame_%04d.png', vf='fps=30').run()

