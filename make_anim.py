import cv2
import os

input_folder = 'smoothed_3d_vis'
output_video = 'EXAMPLE_VIDEO.mp4'

files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])

if not files:
    raise FileNotFoundError(f'There are no PNG files in the {input_folder} folder')

# Read the first frame to get the dimensions
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