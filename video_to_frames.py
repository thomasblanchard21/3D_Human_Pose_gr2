import cv2
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, dest='video')
    args = parser.parse_args()

    return args

# argument parsing
args = parse_args()

def split_video_into_frames(video_path, output_directory):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Read and save frames until the video ends
    frame_count = 0
    while True:
        # Read the next frame from the video
        ret, frame = video.read()

        # Break the loop if no more frames are available
        if not ret:
            break

        # Save the frame as a JPEG image
        frame_path = os.path.join(output_directory, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)

        # Increment the frame count
        frame_count += 1

    # Release the video file
    video.release()

# Example usage
video_path = str(args.video) + '.mp4'
output_directory = "frames"
split_video_into_frames(video_path, output_directory)
