import cv2
import os
import subprocess

def make_frames():
    # Path to video and folder for saving color frames
    VIDEO_PATH = "D:\OrbbecSDK_K4A_Wrapper\include\output_all_color_params_2025-06-27_12-03-41.mkv"  # replace with your .mkv file path
    COLOR_FOLDER = "input_color_frames"

    # Depth frames
    # Even if you change the name, it will still be "input_depth_frames"
    # This variable is only needed for preliminary cleaning of the folder
    DEPTH_FOLDER = "input_depth_frames"

    os.makedirs(COLOR_FOLDER, exist_ok=True)
    os.makedirs(DEPTH_FOLDER, exist_ok=True)

    for filename in os.listdir(COLOR_FOLDER):
        file_path = os.path.join(COLOR_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for filename in os.listdir(DEPTH_FOLDER):
        file_path = os.path.join(DEPTH_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # mkv2depth
    process = subprocess.Popen(["Exe_Helpers\mkv2depth.exe", VIDEO_PATH])

    # mkv2color
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = os.path.join(COLOR_FOLDER, f"color_frame_{frame_idx:05d}.jpg")
        cv2.imwrite(frame_name, frame)
        print(f"JPG file saved: {frame_name}")
        frame_idx += 1
    cap.release()

    process.wait() # wait for mkv2depth
    print("Video parsing complete")

if __name__ == "__main__":
    make_frames()