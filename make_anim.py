import cv2
import os
import re

def make_animation():
    input_folder = 'ekf_3d_vis' # replace with your name folder
    output_video = 'CASE02-E09_2-001-2025-04-25-ANIM.mp4'

    # input_folder = 'experiments/cropped_images' # replace with your name folder
    # output_video = 'CASE02-E06-001-2025-04-25-COLOR.mp4'

    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else -1  # -1 if there is no number (will go to the head of the queue)

    files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith('.jpg')],
        key=extract_number
    )

    if not files:
        raise FileNotFoundError(f'There are no PNG files in the {input_folder} folder')

    first_frame = cv2.imread(os.path.join(input_folder, files[0]))
    height, width, _ = first_frame.shape

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for file in files:
        frame_path = os.path.join(input_folder, file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f'Failed to load {frame_path}, skip...')
            continue
        out.write(frame)

    out.release()
    print(f'Video saved as {output_video}')


if __name__ == "__main__":
    make_animation()  