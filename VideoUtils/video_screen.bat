ffmpeg -f gdigrab -framerate 30 -offset_x 720 -offset_y 240 -video_size 620x720 -i desktop -t 10 output.mp4

Option	Description
-f gdigrab	Screen capture on Windows
-framerate 30	Frame rate (adjust as needed)
-offset_x	X start position of the viewport
-offset_y	Y start position of the viewport
-video_size	Width x Height of the capture area
-i desktop	Capture from the desktop
-t 10   	Capture 10 seconds
output.mp4	Output filename

