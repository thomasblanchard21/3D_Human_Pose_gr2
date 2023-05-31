import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import math
import torch
import json
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.append(".//3DMPPE_ROOTNET_RELEASE//main")
sys.path.append(osp.join('..', '3DMPPE_ROOTNET_RELEASE', 'data'))
sys.path.append(osp.join('..', '3DMPPE_ROOTNET_RELEASE', 'common'))

from config import cfg
from model import get_pose_net
from utils.pose_utils import process_bbox
from dataset import generate_patch_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame', type=str, dest='frame')
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, "Please set proper gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# snapshot load
model_path = './3DMPPE_ROOTNET_RELEASE/output/model_dump/snapshot_%d.pth.tar' % int(args.test_epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
# print('Load checkpoint from {}'.format(model_path))
model = get_pose_net(cfg, False)
if torch.cuda.is_available():
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
else:
    model = DataParallel(model)
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(ckpt['network'])
model.eval()

# prepare input image
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
img_path = str(args.frame)
img_name = img_path.split("/")[-1].split(".")[0]
original_img = cv2.imread(img_path)
original_img_height, original_img_width = original_img.shape[:2]

# prepare bbox for each human
bbox_path = f"bboxes/bbox_output_{img_name}.json"

with open(bbox_path) as f:
    bbox_list = json.load(f)
person_num = len(bbox_list)

# normalized camera intrinsics
focal = [1500, 1500] # x-axis, y-axis
princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis
# print('focal length: (' + str(focal[0]) + ', ' + str(focal[1]) + ')')
# print('principal points: (' + str(princpt[0]) + ', ' + str(princpt[1]) + ')')

# for cropped and resized human image, forward it to RootNet
root_depth_list = []

for n in range(person_num):
    bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
    img, img2bb_trans = generate_patch_image(original_img, bbox, False, 0.0) 
    if torch.cuda.is_available():
        img = transform(img).cuda()[None,:,:,:]
    else:
        img = transform(img)[None,:,:,:]
    k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
    if torch.cuda.is_available():
        k_value = torch.FloatTensor([k_value]).cuda()[None,:]
    else:
        k_value = torch.FloatTensor([k_value])[None,:]

    # forward
    with torch.no_grad():
        root_3d = model(img, k_value) # x,y: pixel, z: root-relative depth (mm)
    img = img[0].cpu().numpy()
    root_3d = root_3d[0].cpu().numpy()

    root_depth_list.append(root_3d[2])
    # print(root_3d)

    # # save output in 2D space (x,y: pixel)
    # vis_img = img.copy()
    # vis_img = vis_img * np.array(cfg.pixel_std).reshape(3,1,1) + np.array(cfg.pixel_mean).reshape(3,1,1)
    # vis_img = vis_img.astype(np.uint8)
    # vis_img = vis_img[::-1, :, :]
    # vis_img = np.transpose(vis_img,(1,2,0)).copy()
    # vis_root = np.zeros((2))
    # vis_root[0] = root_3d[0] / cfg.output_shape[1] * cfg.input_shape[1]
    # vis_root[1] = root_3d[1] / cfg.output_shape[0] * cfg.input_shape[0]
    # cv2.circle(vis_img, (int(vis_root[0]), int(vis_root[1])), radius=5, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)
    # cv2.imwrite('output_root_2d_' + str(n) + '.jpg', vis_img)
    
    # print('Root joint depth: ' + str(root_3d[2]) + ' mm')

# print(root_depth_list)
root_depth_list = [float(x) for x in root_depth_list]

with open(f'3DMPPE_ROOTNET_RELEASE/output/result/root_depth_{img_name}.json', 'w') as f:
    # Use the dump() function to write the list to the file
    json.dump(root_depth_list, f)

