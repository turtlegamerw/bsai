import cv2
import os

# List of video file paths
video_files = ["gameplay.mp4", "gameplay1.mp4", "gameplay2.mp4", "gameplay3.mp4", "gameplay4.mp4", "gameplay5.mp4", "gameplay6.mp4", "gameplay7.mp4", "gameplay8.mp4", "gameplay9.mp4",]

# Output directory for frames
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

# Iterate through video files
for video_file in video_files:
    video = cv2.VideoCapture(video_file)
    success, frame = video.read()
    count = 0

    # Extract video name without extension for folder organization
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # Extract frames
    while success:
        frame_path = os.path.join(video_output_dir, f"{video_name}_frame_{count}.jpg")
        cv2.imwrite(frame_path, frame)
        success, frame = video.read()
        count += 1

    print(f"Extracted {count} frames from {video_file}")

print("Frame extraction completed!")
