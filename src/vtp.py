import cv2

video = cv2.VideoCapture("gameplay.mp4")
success, frame = video.read()
print("succes")
count = 0
while success:
    cv2.imwrite(f"frames/frame_{count}.jpg", frame)
    success, frame = video.read()
    print("succes")
    count += 1
