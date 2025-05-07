@echo off
rm screen_capture.mp4
echo Waiting 3 seconds before starting recording...
ping -n 4 127.0.0.1 >nul

echo Recording screen for 15 seconds...
ffmpeg -f gdigrab -framerate 30 -i desktop -t 15 -c:v libx264 -pix_fmt yuv420p screen_capture.mp4

echo Recording complete! Saved as screen_capture.mp4
exit

