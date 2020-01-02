# --------------------------------------------------------
# Modify based on Tensorflow iCAN
# --------------------------------------------------------

"""
Generating training instances
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import pickle
import random
from random import randint
import tensorflow as tf
import cv2
from config_vcoco import cfg
import ipdb

from utils import *

def encode_boxes(boxes_H, boxes_O):
    # relative spatial encoding
    assert len(boxes_H)==len(boxes_O)
    H_encode = bbox_transform(np.array(boxes_O),np.array(boxes_H))
    O_encode = bbox_transform(np.array(boxes_H),np.array(boxes_O))
    HO_encode = H_encode - O_encode
    
    return H_encode, O_encode, HO_encode

def bb_IOU(boxA, boxB):  #y1 x1 y2 x2

    ixmin = np.maximum(boxA[1], boxB[1])
    iymin = np.maximum(boxA[0], boxB[0])
    ixmax = np.minimum(boxA[3], boxB[3])
    iymax = np.minimum(boxA[2], boxB[2])

    # izmin = np.maximum(boxA[4], boxB[4])
    # izmax = np.minimum(boxA[5], boxB[5])

    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    # it = np.maximum(izmax - izmin + 1., 0.)
    # inters = iw * ih * it
    inters = iw * ih

    # union
    uni = (((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.)) +
          ((boxA[2] - boxA[0] + 1.) * (boxA[3] - boxA[1] + 1.)) - inters)

    overlaps = inters / uni
    return overlaps

def Augmented_box(bbox, shape, image_id, augment = 15):

    thres_ = 0.7

    # box = np.array([0, bbox[0],  bbox[1],  bbox[2],  bbox[3], bbox[4], bbox[5]]).reshape(1,7) #[x1, y1, x2, y2, z1, z2]
    box = np.array([0, bbox[0],  bbox[1],  bbox[2],  bbox[3]]).reshape(1,5) #[y1, x1, y2, x2]
    box = box.astype(np.float64)
        
    count = 0
    time_count = 0
    while count < augment:
        
        time_count += 1
        width = bbox[3] - bbox[1]
        height  = bbox[2] - bbox[0]
        # time = bbox[5] - bbox[4]

        width_cen = (bbox[3] + bbox[1]) / 2
        height_cen  = (bbox[2] + bbox[0]) / 2
        # time_cen = (bbox[5] + bbox[4]) / 2

        ratio = 1 + randint(-10,10) * 0.01

        height_shift = randint(-np.floor(height),np.floor(height)) * 0.1
        width_shift  = randint(-np.floor(width),np.floor(width)) * 0.1
        # time_shift = randint(-np.floor(time),np.floor(time)) * 0.1

        H_1 = max(0, width_cen + width_shift - ratio * width / 2)   # x1
        H_3 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2) # x2
        H_0 = max(0, height_cen + height_shift - ratio * height / 2) # y1
        H_2 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2) # y2

        # H_4 = max(0, time_cen + time_shift - ratio * time / 2)
        # H_5 = min(shape[2] - 1, time_cen + time_shift + ratio * time / 2)
        
        
        if bb_IOU(bbox, np.array([H_0, H_1, H_2, H_3])) > thres_:
            box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1,5)
            box  = np.concatenate((box,     box_),     axis=0)
            count += 1
        if time_count > 150:
            return box
            
    return box

def Generate_activity_CAD120(GT_idx):
    action_ = np.zeros(10)
    action_[int(GT_idx)] = 1
    action_ = action_.reshape(1,10)
    return action_

def Generate_subactivity_human_CAD120(GT_idx):
    action_ = np.zeros(10)
    action_[int(GT_idx)] = 1
    action_ = action_.reshape(1,10)
    return action_

def Generate_subactivity_object_CAD120(GT_idx):
    action_ = np.zeros(12)
    action_[int(GT_idx)] = 1
    action_ = action_.reshape(1,12)
    return action_


subactivities = ['reaching', 'moving', 'pouring', 'eating', 'drinking', 'opening', 'placing', 'closing', 'null', 'cleaning']
affordances = ['movable', 'stationary', 'reachable', 'pourable', 'pourto', 'containable', 'drinkable', 'openable', 'placeable', 'closeable', 'cleanable', 'cleaner']
align_subact_to_aff = [2, 0, 3, 2, 6, 7, 8, 9, 1, 10]

def get_reshaped_pose(feats):
    out_shape = 20
    reshaped = np.zeros([out_shape, 15, 3]).astype('float32')
    if len(feats) < out_shape:
        reshaped[:len(feats)] = feats
    else:
        stride = int(len(feats)/out_shape)
        temp = feats[::stride]
        reshaped = temp[:out_shape]
    return reshaped

def get_reshaped_obj(feats):
    out_shape = 20
    reshaped = np.zeros([out_shape, 10]).astype('float32')
    if len(feats) < out_shape:
        reshaped[:len(feats)] = feats
    else:
        stride = int(len(feats)/out_shape)
        temp = feats[::stride]
        reshaped = temp[:out_shape]
    return reshaped

def Get_Next_Instance_HO_Neg(Trainval_GT, ctrl):#trainval_GT, Trainval_Neg, video_segment_pair, Pos_augment, Neg_select):
    # for each human-object pair sample to form a batch
    # video_id = video_segment_pair[0]
    # GT       = trainval_GT[video_id]['segments'][segment_no]

    # pretrained_dir = "/home/rishabh/scene_graph/hoi_graph/pytorch-i3d/I3D_CAD120_pretrained_features"
    # base_file_name = "I3D_pretrained_" + str(video_id) + "_" + str(segment_no) + "_"
    # human_file = pretrained_dir + "/" + base_file_name + "human.npy"
    # object_file = pretrained_dir + "/" + base_file_name + "object.npy"
    # combined_file = pretrained_dir + "/" + base_file_name + "combined.npy"

    # human_features = np.load(human_file)
    # object_features = np.load(object_file)
    # combined_features = np.load(combined_file)
    video_dir = "/home/rishabh/scene_graph/hoi_graph/dataset/All_subjects_images/"
    pretrained_dir = "/home/rishabh/scene_graph/hoi_graph/pytorch-i3d"
    gpnn_feat_file = pretrained_dir + "/cad120_data.p"
    gpnn_data = pickle.load(open(gpnn_feat_file,'rb'), encoding='latin1')
    
    blobs_multiple = []
    train_size = 100
    if ctrl == 'train':
        keys = list(gpnn_data.keys())[:train_size]
    if ctrl == "test":
        keys = list(gpnn_data.keys())[train_size:]

    video_keys = os.listdir(video_dir)
    for video_id in keys:
        if video_id not in video_keys :
            continue
        frames_dir = os.path.join(video_dir, video_id)
        vid_data = gpnn_data[video_id]
        for seg_num in range(len(vid_data)) :
            if video_id not in Trainval_GT.keys() : 
                print('Video id '+str(video_id)+" not in GT spatial")
                break
            GT_bboxes = Trainval_GT[video_id]
            gt_labels = Trainval_GT[video_id]['gt_labels']
            seg_frames_nums = Trainval_GT[video_id]['frames_nums']
            seg_data = vid_data[seg_num]  
            node_labels = seg_data['node_labels']

            checkLabels = True
            # print('gt_labels', gt_labels)
            # print('node_labels', node_labels)
            if len(gt_labels) != len(node_labels):
                checkLabels = False
            # else :
            #     if subactivities[int(node_labels[0])] != gt_labels[0]:
            #         checkLabels = False
            #     else :
            #         for i in range(1,len(gt_labels)):
            #             if affordances[node_labels[i]] != gt_labels[i]:
            #                 checkLabels = False

            if not checkLabels :
                print('Labels not matching for '+str(video_id)+" "+str(seg_num))
                exit(0) 

            node_features = seg_data['node_features']
            edge_features = seg_data['edge_features']
            for i in range(len(node_labels)-1):
                if node_labels[i+1] == 1 :  
                    if node_labels[0] != 8 :
                        # print('Stationary')
                        continue
                # print('GT', GT)
                action_H, action_O, mask_H, mask_O = Augmented_HO_Neg({'human_affordance': node_labels[0], 'object_affordance': node_labels[i+1] })
                
                # print(edge_features.shape, node_features.shape)
                human_features = np.concatenate((node_features[0], edge_features[0,i+1]))
                object_features = np.concatenate((node_features[i+1], edge_features[0,i+1]))  

                blobs = {}
                blobs['H_num'] = 10
                blobs['human_feat'] = np.reshape(human_features, [1, human_features.shape[0]])
                blobs['object_feat'] = np.reshape(object_features, [1, object_features.shape[0]])

                gt_pose = GT_bboxes['joint_centroids'][seg_num]
                gt_obj_bboxes = GT_bboxes['obj_bboxes'][i][seg_num]


                reshaped_gt_pose = get_reshaped_pose(gt_pose)
                reshaped_gt_obj_bboxes = get_reshaped_obj(gt_obj_bboxes)

                blobs['human_pose'] = reshaped_gt_pose
                blobs['obj_bboxes'] = reshaped_gt_obj_bboxes

                human_joint_bboxes = np.array(gt_pose).astype('float')
                human_bboxes = np.zeros([human_joint_bboxes.shape[0], 4]).astype('float')
                
                x_max = np.max(human_joint_bboxes[:, :, 0], axis=1).astype('float') + 1000.00
                x_min = np.min(human_joint_bboxes[:, :, 0], axis=1).astype('float') + 1000.00
                y_max = np.max(human_joint_bboxes[:, :, 1], axis=1).astype('float') + 1000.00
                y_min = np.min(human_joint_bboxes[:, :, 1], axis=1).astype('float') + 1000.00

                human_bboxes[:, 0] = np.max(np.stack((y_min - 20, np.zeros(y_min.shape[0]).astype('float')), axis=1), axis=1)
                human_bboxes[:, 1] = np.max(np.stack((x_min - 20, np.zeros(x_min.shape[0]).astype('float')), axis=1), axis=1)
                human_bboxes[:, 2] = np.min(np.stack((y_max + 20, np.zeros(y_max.shape[0]).astype('float') + 2000.00), axis=1), axis=1)
                human_bboxes[:, 3] = np.min(np.stack((x_max + 20, np.zeros(x_max.shape[0]).astype('float') + 2000.00), axis=1), axis=1)

                human_bboxes = human_bboxes / 2000.00

                gt_obj_bboxes = np.array(gt_obj_bboxes)
                object_bboxes = np.zeros([len(gt_obj_bboxes), 4]).astype('float')

                object_bboxes[:, 0] = gt_obj_bboxes[:, 1] / 640.00  #y1
                object_bboxes[:, 2] = gt_obj_bboxes[:, 3] / 640.00  #y2
                object_bboxes[:, 1] = gt_obj_bboxes[:, 0] / 480.00  #x1
                object_bboxes[:, 3] = gt_obj_bboxes[:, 2] / 480.00  #x2

                blobs['human_bboxes'] = (human_bboxes * (224 - 1)).astype('int32')
                blobs['object_bboxes'] = (object_bboxes * (224 - 1)).astype('int32')

                blobs['seg_frames'] = seg_frames_nums[seg_num]
                blobs['frames_dir'] = frames_dir

                blobs['gt_class_H']  = action_H
                blobs['gt_class_O']  = action_O
                
                blobs['Mask_H']      = mask_H
                blobs['Mask_O']      = mask_O

                blobs['H_augmented_enc'] = np.reshape(human_features, [1, human_features.shape[0]])
                blobs['O_augmented_enc'] = np.reshape(object_features, [1, object_features.shape[0]])

                blobs_multiple.append(blobs)
    print('Data loading done !')
    return blobs_multiple

def Augmented_HO_Neg(GT):
    # num_joints_coord = len(GT[5])
    # Human_augmented_keypoints_ = GT[5].reshape(1, num_joints_coord)
    action_H_  = Generate_subactivity_human_CAD120(GT['human_affordance'])
    action_O_ =  Generate_subactivity_object_CAD120(GT['object_affordance'])

    mask_H_    = np.ones([1, 10])
    mask_O_    = np.ones([1, 12])
    action_H          = action_H_.reshape( 1, 10 )
    action_O          = action_O_.reshape( 1, 12 )

    mask_H            = mask_H_.reshape( 1, 10 )
    mask_O            = mask_O_.reshape( 1, 12 )
    
    return action_H, action_O, mask_H, mask_O