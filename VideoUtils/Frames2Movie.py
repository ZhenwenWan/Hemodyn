import subprocess

output_video = "output.mp4"
frame_rate = 30  # Adjust as needed

ffmpeg_cmd = [
    "ffmpeg",
    "-r", str(frame_rate),
    "-i", "frames/frame_%04d.png",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    output_video
]

subprocess.run(ffmpeg_cmd)

