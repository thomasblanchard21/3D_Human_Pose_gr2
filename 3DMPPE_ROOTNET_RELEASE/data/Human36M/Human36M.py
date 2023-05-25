import os
import os.path as osp
from pycocotools.coco import COCO
import numpy as np
from config import cfg
from utils.pose_utils import world2cam, cam2pixel, pixel2cam, process_bbox
import cv2
import random
import json
import math

class Human36M:
    def __init__(self, data_split):

        self.camera_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]
        self.data_split = data_split
        self.img_dir = "/work/scitas-share/datasets/Vita/civil-459/h3.6/videos/"
        self.annot_path = osp.join('..', 'data', 'Human36M', 'annotations')
        self.joint_num = 17
        self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.root_idx = self.joints_name.index('Pelvis')
        self.joints_have_depth = True
        self.protocol = 2
        self.data = self.load_data()
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']

    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            if self.protocol == 1:
                subject = [1,5,6,7,8,9]
            elif self.protocol == 2:
                subject = [1,5,6,7,8]
        elif self.data_split == 'test':
            if self.protocol == 1:
                subject = [11]
            elif self.protocol == 2:
                subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject
    
    def load_data(self):

        print('Load data of H36M Protocol ' + str(self.protocol))
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()

        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
        db.createIndex()

        # # extracting images
        # data_file_3d = data_file_3d['positions_3d'].item()
        # n_frame = 0 
        # for s in subject_list:
        #     for a in data_file_3d[s].keys():
        #         n_frame += len(data_file_3d[s][a])  
            

        # all_in_one_dataset_3d = np.zeros((4*n_frame, 17 ,3),  dtype=np.float32)
        # video_and_frame_paths = []
        
        # for s in subject_list:
        #     for a in data_file_3d[s].keys():
        #         print(s,a,len(data_file_3d[s][a]))
        #         for frame_num in range(len(data_file_3d[s][a])):

        #             global_pose = data_file_3d[s][a][frame_num]  
        #             global_pose = global_pose[ self.KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want

        #             for c in range(4):

        #                 tmp = global_pose.copy()

        #                 for j in range(len(tmp)): 
        #                     tmp[j] = tmp[j] - np.divide(np.array(self.camera_parameters[s][c]['translation']),1000)
        #                     tmp[j] = self.qv_mult(np.array(self.camera_parameters[s][c]['orientation']),tmp[j])
                                    
        #                 all_in_one_dataset_3d[i] = tmp  # contains keypoints from all frames

        #                 video_and_frame_paths.append( [self.img_dir+s+"/Videos/"+a+self.camera_ids[c]+".mp4",frame_num]) # contains path to all frames

        #                 i = i + 1 

        if self.data_split == 'test' and not cfg.use_gt_bbox:
            print("Get bounding box from scitas")
            bbox_result = {}
            with open(self.human_bbox_dir) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                bbox_result[str(annot[i]['image_id'])] = np.array(annot[i]['bbox'])
        else:
            print("Get bounding box from groundtruth")

        data = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_width, img_height = img['width'], img['height']

            # check subject and frame_idx
            subject = img['subject']; frame_idx = img['frame_idx'];
            if subject not in subject_list:
                continue
            if frame_idx % sampling_ratio != 0:
                continue

            # camera parameter
            cam_idx = img['cam_idx']
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
                
            # project world coordinate to cam, image coordinate space
            action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx'];
            root_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)[self.root_idx]
            root_cam = world2cam(root_world[None,:], R, t)[0]
            root_img = cam2pixel(root_cam[None,:], f, c)[0]
            joint_vis = np.ones((self.joint_num,1))
            root_vis = np.array(ann['keypoints_vis'])[self.root_idx,None]
            
            # bbox load
            if self.data_split == 'test' and not cfg.use_gt_bbox:
                bbox = bbox_result[str(image_id)]
            else:
                bbox = np.array(ann['bbox'])
            bbox = process_bbox(bbox, img_width, img_height)
            if bbox is None: continue
            area = bbox[2]*bbox[3]

            if subaction_idx == 1:
                img_path = self.img_dir + 'S' + subject + '/outputVideos/' + self.action_name[action_idx-1] + self.camera_ids[cam_idx] + '.mp4/' + '{:04d}'.format(frame_idx+1) + '.jpg'
            elif subaction_idx == 2:
                img_path = self.img_dir + 'S' + subject + '/outputVideos/' + self.action_name[action_idx-1] + ' 1' + self.camera_ids[cam_idx] + '.mp4/' + '{:04d}'.format(frame_idx+1) + '.jpg'
            elif subaction_idx == 3:
                img_path = self.img_dir + 'S' + subject + '/outputVideos/' + self.action_name[action_idx-1] + ' 2' + self.camera_ids[cam_idx] + '.mp4/' + '{:04d}'.format(frame_idx+1) + '.jpg'

            # REMEMBER TO DELETE
            if subject == 1:
                if action_idx == 2:
                    data.append({
                        'img_path': img_path,
                        'img_id': image_id,
                        'bbox': bbox,
                        'area': area,
                        'root_img': root_img, # [org_img_x, org_img_y, depth]
                        'root_cam': root_cam,
                        'root_vis': root_vis,
                        'f': f,
                        'c': c
                    })

        return data

    def evaluate(self, preds, result_dir):
        print('Evaluation start...')
        gts = self.data
        assert len(gts) == len(preds)
        sample_num = len(gts)
 
        pred_save = []
        error = np.zeros((sample_num, 1, 3)) # MRPE
        error_action = [ [] for _ in range(len(self.action_name)) ] # MRPE for each action
        for n in range(sample_num):
            gt = gts[n]
            f = gt['f']
            c = gt['c']
            bbox = gt['bbox']
            gt_root = gt['root_cam']
            
            # warp output to original image space
            pred_root = preds[n]
            pred_root[0] = pred_root[0] / cfg.output_shape[1] * bbox[2] + bbox[0]
            pred_root[1] = pred_root[1] / cfg.output_shape[0] * bbox[3] + bbox[1]
            
            # back-project to camera coordinate space
            pred_root = pixel2cam(pred_root[None,:], f, c)[0]

            # prediction save
            img_id = gt['img_id']
            pred_save.append({'image_id': img_id, 'bbox': bbox.tolist(), 'root_cam': pred_root.tolist()})

            # error calculate
            error[n] = (pred_root - gt_root)**2
            img_name = gt['img_path']
            action_idx = int(img_name[img_name.find('act')+4:img_name.find('act')+6]) - 2
            error_action[action_idx].append(error[n].copy())

        # total error
        tot_err = np.mean(np.sqrt(np.sum(error,axis=2)))
        x_err = np.mean(np.sqrt(error[:,:,0]))
        y_err = np.mean(np.sqrt(error[:,:,1]))
        z_err = np.mean(np.sqrt(error[:,:,2]))
        eval_summary = 'MRPE >> tot: %.2f, x: %.2f, y: %.2f, z: %.2f\n' % (tot_err, x_err, y_err, z_err)
       
        # error for each action
        for i in range(len(error_action)):
            err = np.array(error_action[i])
            err = np.mean(np.power(np.sum(err,axis=2),0.5))
            eval_summary += (self.action_name[i] + ': %.2f ' % err)
        print(eval_summary)

        output_path = osp.join(result_dir, 'bbox_root_human36m_output.json')
        with open(output_path, 'w') as f:
            json.dump(pred_save, f)
        print("Test result is saved at " + output_path)


    # (Quaternion Calculations -> source : https://www.meccanismocomplesso.org/en/hamiltons-quaternions-and-3d-rotation-with-python/ )
    def q_conjugate(self, q):
        w, x, y, z = q
        return (w, -x, -y, -z)

    def q_mult(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return w, x, y, z

    def qv_mult(self, q1, v1):
        # q2 = (0.0,) + v1
        q2 = np.insert(v1,0,0) #new
        return self.q_mult(self.q_mult(q1, q2), self.q_conjugate(q1))[1:]