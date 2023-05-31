#!/bin/bash

# Convert video to frames
python video_to_frames.py --video $1

for frame_file in frames/*.jpg; do
    python get_bbox.py --frame "$frame_file"
    python inference_rootnet.py --frame "$frame_file" --gpu 0 --test_epoch 18
    python inference_posenet_pretrained.py --frame "$frame_file" --gpu 0 --test_epoch 24
done
