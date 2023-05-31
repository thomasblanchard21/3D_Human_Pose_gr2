import cv2
import os

# Path to the directory containing the JPG files
images_folder = '3DMPPE_POSENET_RELEASE/output/vis'

# Output video file path
output_video = 'output_video.mp4'

# Set the desired frame rate (frames per second) for the output video
frame_rate = 24

# Retrieve the list of JPG files in the folder
image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

# Sort the image files in ascending order
image_files.sort()

# Get the dimensions of the first image
first_image = cv2.imread(os.path.join(images_folder, image_files[0]))
height, width, _ = first_image.shape

# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Iterate through the image files and write each frame to the video
for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    frame = cv2.imread(image_path)
    video_writer.write(frame)

# Release the video writer and close the output video file
video_writer.release()

print("Video created successfully at", output_video)
