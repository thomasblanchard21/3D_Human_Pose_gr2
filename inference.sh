#!/bin/bash

python video_to_frames.py --video $1
python get_bbox.py --frame "frames"
python inference_rootnet.py --frame "frames" --gpu 0 --test_epoch 18
python inference_posenet_MuCo.py --frame "frames" --gpu 0 --test_epoch 24
python create_video.py