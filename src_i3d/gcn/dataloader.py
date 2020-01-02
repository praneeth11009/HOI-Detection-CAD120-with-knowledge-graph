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

def gkern3(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    # return kernel
    return kernel / np.sum(kernel)

def Get_Heatmap(coordinates, image_size, sigma=1.0, normalize=False, mean=1.0):
    heatmap = np.zeros([image_size[1], image_size[2]]).astype('float')
    # print(heatmap.shape)
    # print('get_heatmap', coordinates)
    for x in coordinates:
        center_x = int(x[0]*image_size[1])
        center_y = int(x[1]*image_size[2])
        
        kernel = gkern3((6*int(sigma))+1, sigma)
        if normalize:
            kernel = kernel / np.sum(kernel)
        else:
            kernel = kernel*mean

        left_x = max(0, center_x-(3*int(sigma)))
        k_left_x = max(0, -(center_x-(3*int(sigma))))
        right_x = min(image_size[1], center_x+(3*int(sigma))+1)
        k_right_x = max(0, center_x+(3*int(sigma))+1 - image_size[1])
        left_y = max(0, center_y-(3*int(sigma)))
        k_left_y = max(0, -center_y+(3*int(sigma)))
        right_y = min(image_size[2], center_y+(3*int(sigma))+1)
        k_right_y = max(0, center_y+(3*int(sigma))+1 - image_size[2])
        # print('coords', center_x, center_y, left_x, right_x, left_y, right_y)
        # print('kernel', k_left_x, k_right_x, k_left_y, k_right_y)
        # print(heatmap[left_x:right_x, left_y:right_y].shape)
        kernel_size = kernel.shape[0]
        in_shape = kernel[k_left_x:kernel_size-k_right_x, k_left_y:kernel_size-k_right_y].shape
        out_shape = heatmap[left_x:right_x, left_y:right_y].shape

        if (in_shape[0] != out_shape[0] or in_shape[1] != out_shape[1]):
            import pdb
            pdb.set_trace()

        heatmap[left_x:right_x, left_y:right_y] += kernel[k_left_x:kernel_size-k_right_x, k_left_y:kernel_size-k_right_y]

    return heatmap

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

def Generate_activity_CAD120(action_list):
    action_ = np.zeros(10)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,10)
    return action_

def Generate_subactivity_human_CAD120(action_list):
    action_ = np.zeros(11)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,11)
    return action_

def Generate_subactivity_object_CAD120(action_list):
    action_ = np.zeros(13)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,13)
    return action_


def get_reshaped(feats):
    out_shape = 5
    reshaped = np.zeros([out_shape, 7, 7, 1024]).astype('float32')
    if feats.shape[0] < out_shape:
        reshaped[:feats.shape[0]] = feats
    else:
        stride = int(feats.shape[0]/out_shape)
        temp = feats[::stride]
        reshaped[:out_shape] = temp[:out_shape]
    return reshaped
def get_pose_reshaped(feats):
    out_shape = 20
    reshaped = np.zeros([out_shape, 64, 64, 1]).astype('float32')
    if feats.shape[0] < out_shape:
        reshaped[:feats.shape[0]] = feats
    else:
        stride = int(feats.shape[0]/out_shape)
        temp = feats[::stride]
        reshaped = temp[:out_shape]
    return reshaped

def Get_Next_Instance_HO_Neg(trainval_GT, Trainval_Neg, video_segment_pair, Pos_augment, Neg_select):
    # for each human-object pair sample to form a batch
    video_id = video_segment_pair[0]
    segment_no = int(video_segment_pair[1])
    GT       = trainval_GT[video_id]['segments'][segment_no]

    pretrained_dir = "/home/rishabh/scene_graph/hoi_graph/pytorch-i3d/I3D_CAD120_pretrained_features"
    base_file_name = "I3D_pretrained_" + str(video_id) + "_" + str(segment_no) + "_"
    human_file = pretrained_dir + "/" + base_file_name + "human.npy"
    object_file = pretrained_dir + "/" + base_file_name + "object.npy"
    combined_file = pretrained_dir + "/" + base_file_name + "combined.npy"

    human_features = np.load(human_file)
    object_features = np.load(object_file)
    combined_features = np.load(combined_file)
    # video_dir = '/home/nilay/hoi_graph/src_3D/gcn/CAD120/Subject1/rgbd_images/arranging_objects'
    
    # if Trainval_Neg:
    #     # im_file  = cfg.DATA_DIR + '/' + 'v-coco/images/traintest2017/' + (str(image_id)).zfill(12) + '.jpg'
    #     im_dir  =  video_dir + '/' + str(video_id)
    # else:
    #     im_dir  =  video_dir + '/' + str(video_id)
    
    # if not os.path.exists(im_dir):
    #     if Trainval_Neg:
    #         print(iter, im_dir)
    #     else:
    #         blobs = {}
    #         return blobs
    
    # start_frame = GT['start']
    # end_frame = GT['end']

    # image_stack = None
    # for image_ids in range(start_frame, end_frame+1):
    #     im       = cv2.imread(im_dir + '/RGB_' + str(image_ids) + '.png')
    #     im_orig  = im.astype(np.float32, copy=True)
    #     im_orig -= cfg.PIXEL_MEANS
    #     im_shape = im_orig.shape
    #     im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    #     if image_stack is None:
    #         image_stack = im_orig
    #     else:
    #         image_stack = np.concatenate((image_stack, im_orig), axis=0)

    # image_stack_shape = image_stack.shape
    high_level_activity = trainval_GT[video_id]['high_level_activity']
    H_augmented, H_augmented_solo, O_augmented, H_pose_heat_maps, action_HO, action_H, action_O, mask_HO, mask_H, mask_O, H_aug_enc, O_aug_enc, HO_aug_enc = Augmented_HO_Neg(GT, high_level_activity, Trainval_Neg, Pos_augment, Neg_select)
    
    num_images = GT['end'] - GT['start'] + 1
    
    blobs = {}

    blobs['human_feat'] = get_reshaped(human_features)
    blobs['object_feat'] = get_reshaped(object_features)
    blobs['combined_feat'] = get_reshaped(combined_features)
    # print('heatmaps', H_pose_heat_maps.shape)
    pose_reshaped = get_pose_reshaped(H_pose_heat_maps)
    # print('pose_reshaped', pose_reshaped.shape)
    blobs['human_pose_heat_maps'] = pose_reshaped
    # blobs['image']       = image_stack
    # blobs['H_boxes_solo']= H_augmented_solo
    # blobs['H_boxes']     = H_augmented
    # blobs['O_boxes']     = O_augmented

    # blobs['H_poses']     = Human_augmented_keypoints

    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_H']  = action_H
    blobs['gt_class_O']  = action_O
    
    blobs['Mask_HO']     = mask_HO
    blobs['Mask_H']      = mask_H
    blobs['Mask_O']      = mask_O

    blobs['H_num']       = len(action_H)
    blobs['H_augmented_enc']  = H_aug_enc
    blobs['O_augmented_enc']  = O_aug_enc 
    blobs['HO_augmented_enc'] = HO_aug_enc

    return blobs

def Augmented_HO_Neg(GT, high_level_activity, Trainval_Neg, Pos_augment, Neg_select):
    num_images = GT['end'] - GT['start'] + 1
    # print(num_images)
    Human    = GT['human_bboxes'] # simply agent bbox
    Object   = GT['object_bboxes']
    Human_pose = GT['human_joint_bboxes_norm']

    assert(num_images == Human.shape[0])
    assert(num_images == Object.shape[0])
    assert(num_images == Human_pose.shape[0])


    Human_pose_heat_maps = np.zeros([num_images, 64, 64, 1]).astype('float32')
    # Intially Human pose has mean 0 and normalized (sum of squares is 15)
    Human_pose /= np.sqrt(15.00)
    # Now human pose has values between [-1, 1], mean 0 and sum of squares is 1
    Human_pose += 1.000
    Human_pose /= 2.000
    # Now the mean would be at [0.5, 0.5] and sum of squares will be (N+1)/4 = 16/4 = 4
    # All values now in [0, 1]
    for i in range(num_images):
        Human_pose_heat_maps[i, :, :, 0] =  Get_Heatmap(Human_pose[i], np.array([1, 64, 64]), sigma=2.0, normalize=False, mean=1.0)

    # num_joints_coord = len(GT[5])
    # Human_augmented_keypoints_ = GT[5].reshape(1, num_joints_coord)

    action_HO_ = Generate_activity_CAD120([high_level_activity])
    action_H_  = Generate_subactivity_human_CAD120([GT['human_affordance']])
    action_O_ =  Generate_subactivity_object_CAD120([GT['object_affordance']])

    mask_HO_   = np.ones([1, 10])
    mask_H_    = np.ones([1, 11])
    mask_O_    = np.ones([1, 13])
    
    if not Trainval_Neg: # Testval no augment and negative sampling
        # Human_augmented = np.array([Human[0], Human[1], Human[2], Human[3], Human[4]]).reshape(num_images, 5)
        Human_augmented = Human
        Human_augmented.astype(np.float64)
        Object_augmented = Object
        Object_augmented.astype(np.float64)
        Human_augmented_solo = Human_augmented.copy()
        
        H_encode, O_encode, HO_encode = encode_boxes(Human_augmented, Object_augmented)
        
        Human_augmented   = Human_augmented.reshape( num_images, 4 )
        Human_augmented_solo = Human_augmented_solo.reshape( num_images, 4 ) 
        Object_augmented  = Object_augmented.reshape( num_images, 4 ) 

        # Human_augmented_keypoints = Human_augmented_keypoints_.reshape(num, num_joints_coord)
        
        action_HO         = action_HO_.reshape(1, 10 ) 
        action_H          = action_H_.reshape( 1, 11 )
        action_O          = action_O_.reshape( 1, 13 )

        mask_HO           = mask_HO_.reshape(  1, 10 )
        mask_H            = mask_H_.reshape(   1, 11 )
        mask_O            = mask_O_.reshape(   1, 13 )

        H_augmented_encode  = H_encode.reshape( num_images, 4 )
        O_augmented_encode  = O_encode.reshape( num_images, 4 )
        HO_augmented_encode = HO_encode.reshape( num_images, 4 )
        
    else:
        pass
        # Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
        # Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
        # Human_augmented_solo = Human_augmented.copy() # only positive

        # Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
        # Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]


        # num_pos = len(Human_augmented)

        # # Human_augmented_keypoints = Human_augmented_keypoints_
        # # for i in range(num_pos - 1):
        # #     Human_augmented_keypoints = np.concatenate((Human_augmented_keypoints, Human_augmented_keypoints_), axis=0)


        # if image_id in Trainval_Neg:
        #     if len(Trainval_Neg[image_id]) < Neg_select:
        #         for Neg in Trainval_Neg[image_id]:
        #             Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
        #             Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        #             # Human_augmented_keypoints = np.concatenate((Human_augmented_keypoints, Neg[-1].reshape(1, num_joints_coord)), axis=0)
        #     else:
        #         List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
        #         for i in range(Neg_select):
        #             Neg = Trainval_Neg[image_id][List[i]]
        #             Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
        #             Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        #             # Human_augmented_keypoints = np.concatenate((Human_augmented_keypoints, Neg[-1].reshape(1, num_joints_coord)), axis=0)

        # num_pos_neg = len(Human_augmented) # <=46


        # action_HO = action_HO_
        # action_H  = action_H_
        # action_O  = action_O_

        # for i in range(num_pos - 1):
        #     action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        #     action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        #     action_O  = np.concatenate((action_O, action_O_),   axis=0)

        #     # mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

        # # for i in range(num_pos_neg - 1):
        # #     mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

        # for i in range(num_pos_neg - num_pos):
        #     action_HO = np.concatenate((action_HO, np.zeros(10).reshape(1, 10)), axis=0)

        # H_encode, O_encode, HO_encode = encode_boxes(Human_augmented[:,1:], Object_augmented[:,1:])

        # Human_augmented   = Human_augmented.reshape( num_pos_neg, 7)
        # Human_augmented_solo = Human_augmented_solo.reshape( num_pos, 7) 
        # Object_augmented  = Object_augmented.reshape(num_pos_neg, 7) 

        # # Human_augmented_keypoints = Human_augmented_keypoints.reshape(num_pos_neg, num_joints_coord)

        # action_HO         = action_HO.reshape(num_pos_neg, 10) 
        # action_H          = action_H.reshape( num_pos, 10)
        # action_O          = action_O.reshape( num_pos, 12)

        # # mask_HO           = mask_HO.reshape(  num_pos_neg, 26)
        # # mask_H            = mask_H.reshape(   num_pos, 26)

        # H_augmented_encode  = H_encode.reshape( num_pos_neg, 6)
        # O_augmented_encode  = O_encode.reshape( num_pos_neg, 6)
        # HO_augmented_encode = HO_encode.reshape(num_pos_neg, 6)
    
    return Human_augmented, Human_augmented_solo, Object_augmented, Human_pose_heat_maps, \
    action_HO, action_H, action_O, mask_HO, mask_H, mask_O, H_augmented_encode,O_augmented_encode, HO_augmented_encode