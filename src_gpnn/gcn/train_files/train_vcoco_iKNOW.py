from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import os.path
import sys
import importlib

root = '/home/rishabh/scene_graph/hoi_graph'
sys.path.append(root+'/src2/gcn/')
import tensorflow as tf
import pickle as pkl
import numpy as np
import networkx as nx
import json
import h5py
import ipdb
import scipy.io as sio
import argparse

from utils import *
from dataloader import *
from config_vcoco import cfg
sys.path.append(root+'/src2/gcn/test_files/')

def from_snapshot(sess):
    
    if args.Restore_flag == 0:

        saver_t  = [var for var in tf.model_variables() if 'conv1' in var.name and 'conv1_sp' not in var.name]
        saver_t += [var for var in tf.model_variables() if 'conv2' in var.name and 'conv2_sp' not in var.name]
        saver_t += [var for var in tf.model_variables() if 'conv3' in var.name]
        saver_t += [var for var in tf.model_variables() if 'conv4' in var.name]
        saver_t += [var for var in tf.model_variables() if 'conv5' in var.name]
        saver_t += [var for var in tf.model_variables() if 'shortcut' in var.name]

        sess.run(tf.global_variables_initializer())

        print('Restoring model snapshots from {:s}'.format(pretrained_model))

        saver_restore = tf.train.Saver(saver_t)
        saver_restore.restore(sess, pretrained_model)
        

    if args.Restore_flag == 5 or args.Restore_flag == 6 or args.Restore_flag == 7:

        sess.run(tf.global_variables_initializer())        
            
        print('Restoring model snapshots from {:s}'.format(pretrained_model))
        saver_t = {}
            
        # Add block0
        for ele in tf.model_variables():
            if 'resnet_v1_50/conv1/weights' in ele.name or 'resnet_v1_50/conv1/BatchNorm/beta' in ele.name or 'resnet_v1_50/conv1/BatchNorm/gamma' in ele.name or 'resnet_v1_50/conv1/BatchNorm/moving_mean' in ele.name or 'resnet_v1_50/conv1/BatchNorm/moving_variance' in ele.name:
                saver_t[ele.name[:-2]] = ele
            # Add block1
        for ele in tf.model_variables():
            if 'block1' in ele.name:
                saver_t[ele.name[:-2]] = ele
           
        # Add block2
        for ele in tf.model_variables():
            if 'block2' in ele.name:
                saver_t[ele.name[:-2]] = ele
                    
        # Add block3
        for ele in tf.model_variables():
            if 'block3' in ele.name:
                saver_t[ele.name[:-2]] = ele
                
        # Add block4
        for ele in tf.model_variables():
            if 'block4' in ele.name:
                saver_t[ele.name[:-2]] = ele
            
        saver_restore = tf.train.Saver(saver_t)
        saver_restore.restore(sess, pretrained_model)
            
    if args.Restore_flag >= 5:
        saver_t = {}
        # Add block5
        for ele in tf.model_variables():
            if 'block4' in ele.name:
                saver_t[ele.name[:-2]] = [var for var in tf.model_variables() if ele.name[:-2].replace('block4','block5') in var.name][0]
        saver_restore = tf.train.Saver(saver_t)
        saver_restore.restore(sess, pretrained_model)

    if args.Restore_flag >= 6:
        saver_t = {}
        # Add block6
        for ele in tf.model_variables():
            if 'block4' in ele.name:
                saver_t[ele.name[:-2]] = [var for var in tf.model_variables() if ele.name[:-2].replace('block4','block6') in var.name][0]
        saver_restore = tf.train.Saver(saver_t)
        saver_restore.restore(sess, pretrained_model)
                
    if args.Restore_flag >= 7:
        saver_t = {}
        # Add block7
        for ele in tf.model_variables():
            if 'block4' in ele.name:
                saver_t[ele.name[:-2]] = [var for var in tf.model_variables() if ele.name[:-2].replace('block4','block7') in var.name][0]
        saver_restore = tf.train.Saver(saver_t)
        saver_restore.restore(sess, pretrained_model)
        
parser = argparse.ArgumentParser(description='Training Parameters')
parser.add_argument('--sim_weight', type=float, default=0.8)
parser.add_argument('--model', dest='model',
                    help='Select model',
                    default='Triple_GCN', type=str)
parser.add_argument('--Restore_flag', dest='Restore_flag',
                    help='Number of Res5 blocks',
                    default=4, type=int)
parser.add_argument('--Pos_augment', dest='Pos_augment', 
                    help='Number of augmented detection for each one. (By jittering the object detections)',
                    default=15, type=int)
parser.add_argument('--Neg_select', dest='Neg_select', 
                    help='Number of Negative example selected for each image',
                    default=30, type=int)
parser.add_argument('--gpu', dest='gpu', help='Specify GPU(s)',
                    default='0 1', type=str)
args = parser.parse_args()

flags = tf.app.flags 
FLAGS = flags.FLAGS 
flags.DEFINE_float('sim_weight', args.sim_weight, 'similarity weight')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')

if args.model == 'Triple_GCN':
    from iKNOW_VCOCO import Triple_GCN
    model_func = Triple_GCN
    modelname = 'Triple_GCN'.lower()

is_training = True 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(cfg.RNG_SEED)

# data_dir = root + '/src2/gcn/CAD120/All_subjects_images'
output_dir = root + '/output/CAD120'
# vcoco_dir = data_dir + 'v-coco'
# dataset = data_dir + 'glove_vcoco_vrd'
data_dir = root + '/data/'
# dataset = data_dir + 'glove_cad120'
dataset = data_dir + 'glove_cad120_10'
pretrained_model = root + '/weights/res50_faster_rcnn/res50_faster_rcnn_iter_1190000.ckpt'

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-%H-%M')
# timestamp = '11-17-03-14'
savepath = os.path.join(output_dir, "CAD120_models", modelname+'-%s'%timestamp)
# savepath = os.path.join(output_dir, os.path.basename(dataset), "triple_gcn-09-04-13-15")
savename = modelname+'-%s'%timestamp
if not os.path.exists(savepath):
    os.makedirs(savepath)   

    
print('==> Load data')
wordlist_file = os.path.join(data_dir, 'list/words_cad120_10.json')
    
with open(wordlist_file) as fp:
    wordlist = json.load(fp)
# meta = h5py.File(vcoco_dir + '/devkit/vcoco_meta.h5','r')

# # train_file = vcoco_dir + '/mydata/annotation/trainval.json'
# train_file = vcoco_dir + '/mydata/trainval.json'

# with open(train_file) as json_data:
#     rdata_train = json.load(json_data)
# # test_file = vcoco_dir + '/mydata/annotation/test.json'
# test_file = vcoco_dir + '/mydata/test.json'
# with open(test_file) as json_data:
#     rdata_test = json.load(json_data)

Trainval_GT = pkl.load( open( root + '/src2/gcn/CAD120_GT_all_subjects_training_new_wo_having_meal.pkl', "rb" ), encoding='latin1' )
Testval_GT = pkl.load( open( root + '/src2/gcn/CAD120_GT_all_subjects_testing_new_wo_having_meal.pkl', "rb" ), encoding='latin1' )

iv_all = []
for i in range(10, len(wordlist)):
    # word = meta['meta/pre/idx2name/' + str(i)][...]
    # iv_all.append(wordlist.index(str(word)[2:-1]))
    iv_all.append(i)

features, support = load_graph(data_dir, os.path.basename(dataset))

print('==> Create model and initialize')
model = model_func(is_training=is_training) 

# sess = tf.Session(config=create_config_proto())
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True

# debug variables
total_var = tf.trainable_variables()
            
# resume training or initialize 
# savepath = savepath + '/bestckpt'
ckptpath = savepath + '/checkpoint' 
loadpath = savepath
if False and os.path.exists(ckptpath):
    # model.load_best(sess, savepath)
    # from_snapshot(sess)  # res50 5 blocks pretrained_model
    model.resume(sess, loadpath) 
    print("Resuming")
    # assert(1==0)
else:
    from_snapshot(sess)  # res50 5 blocks pretrained_model

# log
merge = tf.summary.merge_all()

print('==> Start training')
loss_save = [] 
loss_save.append(np.inf)
loss_train = 0.0
count_val = 0
saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess.run(init)

# error_segments_list = pkl.load( open( root + '/pytorch-i3d/error_segments_list.pkl', "rb" ), encoding='latin1' )
# error_segments_list = set(error_segments_list)

# print(error_segments_list)

GPUs = ['/device:gpu:0', '/device:gpu:1']
Towers = ['tower_0','tower_1'] 

log_train_file = open("log_train_file_gpnn", "a+")
log_test_file = open("log_test_file_gpnn", "a+")
res_log_file = open("res_log_gpnn", "w+")

blobs_multiple_train = Get_Next_Instance_HO_Neg(ctrl = 'train')
blobs_multiple_test = Get_Next_Instance_HO_Neg(ctrl = 'test')

for epoch in range(cfg.TRAIN.EPOCHS + 1): 
    timer = Timer()
    timer.tic()
    human_subactivity_acc = 0.
    object_subactivity_acc = 0.
    # combined_subactivity_acc = 0.
    
    log_train_file = open("log_train_file_gpnn", "a")
    log_test_file = open("log_test_file_gpnn", "a")
    res_log_file = open("res_log_gpnn", "a")
    res_log_file.write("Epoch "+str(epoch)+"\n")

    final_data_len = len(blobs_multiple_train)
    loss_train = 0.0

    for blobs in blobs_multiple_train:
        blobs['support'] = support
        blobs['features'] = features
        blobs['num_features_nonzero'] = features[1].shape
        blobs['iv_all'] = np.array(iv_all)

        # print(np.array(support).shape)
        # print()
        # print(features.shape)
        # print(iv_all)

        train_flag_used, summary, predictions, total_loss, now_lr, sess = model.train_step_with_summary(sess, blobs, merge) 
        loss_train += total_loss
        assert(train_flag_used == True)

        ####################
        H_pred  = np.argmax(predictions["cls_score_H"])
        O_pred  = np.argmax(predictions["cls_score_O"])
        # HO_pred  = np.argmax(predictions["cls_score_HO"])

        label_H = np.argmax(blobs['gt_class_H'])
        # label_HO = np.argmax(blobs['gt_class_HO'])
        label_O = np.argmax(blobs['gt_class_O'])

        res_log_file.write('human '+str(label_H)+" "+str(H_pred)+"\n")
        res_log_file.write('object '+str(label_O)+" "+str(O_pred)+"\n")

        if label_H == H_pred:
            human_subactivity_acc += 1.0

        if label_O == O_pred:
            object_subactivity_acc += 1.0

        # if label_HO == HO_pred:
        #     combined_subactivity_acc += 1.0
        ####################
        # assert(human_subactivity_acc <= Data_length)
        # assert(combined_subactivity_acc <= Data_length)
        # assert(object_subactivity_acc <= Data_length)
        

    human_subactivity_acc = (human_subactivity_acc / float(final_data_len)) * 100.00
    object_subactivity_acc = (object_subactivity_acc / float(final_data_len)) * 100.00
    # combined_subactivity_acc = (combined_subactivity_acc / float(Data_length - len_invalid_batches)) * 100.00

    timer.toc()
    if epoch % 1 == 0: # every 10 epochs            
        print('Epoch:%02d, total loss: %.6f, human_acc: %.6f, object_acc: %.6f, lr: %f, speed: %.3f s/iter' %  
              (epoch, loss_train/final_data_len, human_subactivity_acc, object_subactivity_acc, now_lr, timer.average_time))
        log_train_file.write('Epoch:%02d, total loss: %.6f, human_acc: %.6f, object_acc: %.6f, lr: %f, speed: %.3f s/iter\n' %  
              (epoch, loss_train/final_data_len, human_subactivity_acc, object_subactivity_acc, now_lr, timer.average_time))
    # every epoch validation
    
    loss_val = 0.0 

    test_human_subactivity_acc = 0.
    test_object_subactivity_acc = 0.
    # test_combined_subactivity_acc = 0. 

    timer.tic()  
    
    # for test_batch in test_batch_ids: 

        # if test_batches[test_batch] in error_segments_list:
        #     test_len_invalid_batches += 1.
        #     continue

    # blobs_multiple = Get_Next_Instance_HO_Neg(Testval_GT, None, test_batches[test_batch], None, None)
    test_final_data_len = len(blobs_multiple_test)

    for blobs in blobs_multiple_test:
        blobs['support'] = support
        blobs['features'] = features
        blobs['num_features_nonzero'] = features[1].shape
        blobs['iv_all'] = np.array(iv_all)

        train_flag, summary, predictions, total_loss, sess = model.test_step_with_summary(sess, blobs, merge) 
        loss_val = loss_val + total_loss
        assert(train_flag==False)

        ####################
        test_H_pred  = np.argmax(predictions["cls_score_H"])
        test_O_pred  = np.argmax(predictions["cls_score_O"])
        # test_HO_pred  = np.argmax(predictions["cls_score_HO"])

        test_label_H = np.argmax(blobs['gt_class_H'])
        # test_label_HO = np.argmax(blobs['gt_class_HO'])
        test_label_O = np.argmax(blobs['gt_class_O'])

        if test_label_H == test_H_pred:
            test_human_subactivity_acc += 1.0

        if test_label_O == test_O_pred:
            test_object_subactivity_acc += 1.0

        # if test_label_HO == test_HO_pred:
        #     test_combined_subactivity_acc += 1.0
        #################### 

    test_human_subactivity_acc = (test_human_subactivity_acc / float(test_final_data_len)) * 100.00
    test_object_subactivity_acc = (test_object_subactivity_acc / float(test_final_data_len)) * 100.00
    # test_combined_subactivity_acc = (test_combined_subactivity_acc / float(test_final_data_len)) * 100.00     

    timer.toc()

    loss_val = loss_val/test_final_data_len
    print('Epoch:%02d'%(epoch) + ', time={:.3f} s/iter'.format(timer.average_time) + \
    	', eval_loss={:.5f}'.format(loss_val) + ', test_final_data_len=%02d'%(test_final_data_len) + \
    	', len_val=%02d'%(test_final_data_len) + ', test_human_subactivity_acc: %.6f'%(test_human_subactivity_acc) + \
    	', test_object_subactivity_acc: %.6f'%(test_object_subactivity_acc)) 
    log_test_file.write('Epoch:%02d'%(epoch) + ', time={:.3f} s/iter'.format(timer.average_time) + \
    	', eval_loss={:.5f}'.format(loss_val) + ', test_final_data_len=%02d'%(test_final_data_len) + \
    	', len_val=%02d'%(test_final_data_len) + ', test_human_subactivity_acc: %.6f'%(test_human_subactivity_acc) + \
    	', test_object_subactivity_acc: %.6f\n'%(test_object_subactivity_acc))
    log_train_file.close()
    res_log_file.close()
    log_test_file.close()
    if loss_val<loss_save[-1]:
        loss_save.append(loss_val)
        model.save_best(sess, savepath) 
        print(loss_save)
                
    save_path = savepath + "/%s_latest.ckpt" % modelname
    saver.save(sess, save_path)
    print("Latest model saved in file: %s" % save_path)       

sess.close()
log_train_file.close()
log_test_file.close()