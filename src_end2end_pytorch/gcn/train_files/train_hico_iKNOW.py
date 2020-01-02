from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import datetime
import os
import os.path
import sys
root = '/home/hoi_graph'
sys.path.append(root+'/src/gcn/')
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
from config_hico import cfg

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

        
def train_net(sess, epoch, count_train, count_val, train_writer, val_writer, loss_save):
    timer = Timer()
   
    Data_length = len(Trainval_GT)
    loss_train = 0.0 
    iter = 0    
    while iter < Data_length + 1: 
        timer.tic()
        
        iter +=1
        count_train +=1
        blobs = Get_Next_Instance_HO_Neg_HICO(Trainval_GT, Trainval_N, iter, args.Pos_augment, args.Neg_select, Data_length)
            
        blobs['support'] = support
        blobs['features'] = features
        blobs['num_features_nonzero'] = features[1].shape
        blobs['iv_all'] = np.array(iv_all)
        
        summary, total_loss, now_lr, sess = model.train_step_with_summary(sess, blobs, merge) 
        loss_train += total_loss
        
        train_writer.add_summary(summary, count_train) 
        
        timer.toc()
            
        if iter % (cfg.TRAIN.DISPLAY) == 0:
            
            print('Epoch:%02d, iter: %d / %d, im_id: %u, total loss: %.6f, lr: %f, speed: %.3f s/iter' %  
                  (epoch, iter, Data_length, Trainval_GT[iter%Data_length][0], loss_train/iter, now_lr, timer.average_time))
                
        if (iter % cfg.TRAIN.SNAPSHOT_ITERS == 0 and iter != 0) or (iter == 10):
            
            loss_val, dur_val, count_val, val_writer = val_net(sess, count_val, val_writer)          
            print('Epoch:%02d'%(epoch) + ', time={:.3f} s/iter'.format(dur_val) + ', eval_loss={:.5f}'.format(loss_val)) 

            if loss_val<loss_save[-1]:
                loss_save.append(loss_val)
                model.save_best(sess, savepath)
    
    return sess, count_train, count_val, train_writer, val_writer, loss_save      


def val_net(sess, count_val, val_writer):
    timer = Timer()
    
    Data_length = len(Testval_GT)
    loss_val = 0.0 
    iter_val = 0    
    while iter_val < Data_length + 1: 
        timer.tic()
        
        iter_val +=1
        blobs = Get_Next_Instance_HO_Neg_HICO(Testval_GT, None, iter_val, None, None, Data_length)
            
        blobs['support'] = support
        blobs['features'] = features
        blobs['num_features_nonzero'] = features[1].shape
        blobs['iv_all'] = np.array(iv_all)
        
        if count_val>-1:
            count_val +=1
            summary, total_loss = model.test_step_with_summary(sess, blobs, merge) 
            loss_val = loss_val + total_loss      

            val_writer.add_summary(summary, count_val)  
        else:
            total_loss = model.test_step(sess, blobs) 
            loss_val = loss_val + total_loss
            val_writer = None

        timer.toc()
            
    loss_val = loss_val/iter_val
        
    return loss_val, timer.average_time, count_val, val_writer


    
parser = argparse.ArgumentParser(description='Training Parameters')
parser.add_argument('--sim_weight', type=float, default=0.8)
parser.add_argument('--model', dest='model',
                    help='Select model',
                    default='Triple_GCN', type=str)
parser.add_argument('--Restore_flag', dest='Restore_flag',
                    help='Number of Res5 blocks',
                    default=5, type=int)
parser.add_argument('--Pos_augment', dest='Pos_augment', 
                    help='Number of augmented detection for each one. (By jittering the object detections)',
                    default=15, type=int)
parser.add_argument('--Neg_select', dest='Neg_select', 
                    help='Number of Negative example selected for each image',
                    default=30, type=int)
parser.add_argument('--gpu', dest='gpu', help='Specify GPU(s)',
                    default='2', type=str)
args = parser.parse_args()

flags = tf.app.flags 
FLAGS = flags.FLAGS 
flags.DEFINE_float('sim_weight', args.sim_weight, 'similarity weight')

if args.model == 'Triple_GCN':
    from iKNOW_HICO import Triple_GCN
    model_func = Triple_GCN
    modelname = 'Triple_GCN'.lower()
    
is_training = True 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(cfg.RNG_SEED)

data_dir = root + '/data/'
output_dir = root + '/output'
hico_dir = data_dir + 'hico_20160224_det'
dataset = data_dir + 'glove_hico_vrd'
pretrained_model = root + '/Weights/res50_faster_rcnn/res50_faster_rcnn_iter_1190000.ckpt'

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-%H-%M')
savepath = os.path.join(output_dir, os.path.basename(dataset), modelname+'-%s'%timestamp)
if not os.path.exists(savepath):
    os.makedirs(savepath)   

    
print('==> Load data')
wordlist_file = os.path.join(data_dir, 'list/words_hico_vrd.json')
    
with open(wordlist_file) as fp:
    wordlist = json.load(fp)
meta = h5py.File(os.path.join(hico_dir, 'devkit/hico_det_meta.h5'),'r')

anno_file = os.path.join(hico_dir, 'devkit/anno_bbox.mat')
anno = sio.loadmat(anno_file, struct_as_record=False, squeeze_me=True)
list_action = anno['list_action']

Trainval_GT = pkl.load( open( data_dir + 'Trainval_GT_HICO.pkl', "rb" ) )
Trainval_N  = pkl.load( open( data_dir + 'Trainval_Neg_HICO.pkl', "rb" ) )
Testval_GT  = pkl.load( open( data_dir + 'Testval_GT_HICO.pkl', "rb" ) )

iv_all = []
for i in range(117):
    word = meta['meta/pre/idx2name/' + str(i)][...]
    iv_all.append(wordlist.index(str(word)))

features, support = load_graph(data_dir, os.path.basename(dataset))

print('==> Create model and initialize')
model = model_func(is_training=is_training) 

sess = tf.Session(config=create_config_proto())

# debug variables
total_var = tf.trainable_variables()
for var in total_var:
    print(var)
            
# resume training or initialize 
ckptpath = savepath + '/checkpoint' 
loadpath = savepath
if os.path.exists(ckptpath):
    model.resume(sess, loadpath) 
else:
    from_snapshot(sess)  # res50 5 blocks pretrained_model

# log
train_writer = tf.summary.FileWriter(root+'/log/hico/%s-%s/'%(model.name, timestamp)+'plot_train', sess.graph) 
val_writer = tf.summary.FileWriter(root+'/log/hico/%s-%s/'%(model.name, timestamp)+'plot_val')
merge = tf.summary.merge_all()

print('==> Start training')
loss_save = [] 
loss_save.append(np.inf)
count_train = 0
count_val = 0
saver = tf.train.Saver() 
for epoch in range(cfg.TRAIN.EPOCHS + 1): 
 
    sess, count_train, count_val, train_writer, val_writer, loss_save = train_net(sess,epoch, count_train, count_val, train_writer, val_writer, loss_save)   

    save_path = savepath + "/%s_latest.ckpt" % modelname
    saver.save(sess, save_path)
    print("Latest model saved in file: %s" % save_path)     
        
print('==> Start testing')
model.load_best(sess, savepath)
loss_test, dur_test, count_test, _ = val_net(sess, -1, None) 
print('Best cost={:.5f}'.format(loss_test))

sess.close()
train_writer.close()
val_writer.close()


