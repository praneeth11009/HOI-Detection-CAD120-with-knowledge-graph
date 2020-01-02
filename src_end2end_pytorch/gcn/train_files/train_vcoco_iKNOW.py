import time
import datetime
import os
import os.path
import sys

root = '/home/rishabh/scene_graph/hoi_graph'
sys.path.append(root+'/src_py/gcn/')
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import numpy as np
import time
import h5py
import cv2
import shutil
import argparse
from config_vcoco import cfg
from utils import *

from dataloader import *
sys.path.append(root+'/src_py/gcn/')

from iKNOW_VCOCO import Triple_GCN

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
                    default='1', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

output_dir = root + '/output/CAD120'
data_dir = root + '/data/'
dataset = data_dir + 'glove_cad120_10'

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-%H-%M')
savepath = os.path.join(output_dir, "CAD120_models/") 

print('==> Load data')
wordlist_file = os.path.join(data_dir, 'list/words_cad120_10.json')
with open(wordlist_file) as fp:
    wordlist = json.load(fp)

iv_all = []
for i in range(10, len(wordlist)):
    iv_all.append(i)

features, support = load_graph(data_dir, os.path.basename(dataset))

print('==> Create model and initialize')
learning_rate = cfg.TRAIN.LEARNING_RATE
load_checkpoint = False
TRAIN_EPOCHS = cfg.TRAIN.EPOCHS
SCHEDULER_STEP = cfg.TRAIN.STEPSIZE
gamma = cfg.TRAIN.GAMMA
momentum = cfg.TRAIN.MOMENTUM

# mode = 'i3d'
mode = 'spatial'
# mode = 'i3d_spatial'


model = Triple_GCN(is_training=True,mode=mode)
model.float().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
lambda1 = lambda epoch: gamma
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

if load_checkpoint:
    checkpoint_file = savepath + 'tripleGCN_'+mode+'_model_best.pth'
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'learning_rate' in list(checkpoint.keys()):
          learning_rate = checkpoint['learning_rate']
    else:
        print('Checkpoint file does not exist, train from start')


def save_checkpoint(state, is_best, filename=savepath + 'tripleGCN_'+mode+'_model_checkpoint.pth'):
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, savepath + 'tripleGCN_'+mode+'_model_best.pth')

print('==> Start training')
print('Learning Rate, gamma', learning_rate, gamma)
loss_save = [] 
loss_save.append(np.inf)
count_val = 0

log_train_file = open("log_train_file_"+mode+'_2', "a+")
log_test_file = open("log_test_file_"+mode+'_2', "a+")
# res_log_file = open("res_log", "w+")

Trainval_GT = pkl.load( open( root + '/src_py/gcn/cad120_gt_spatial.p', "rb" ), encoding='latin1' )

blobs_multiple_train = Get_Next_Instance_HO_Neg(Trainval_GT, ctrl = 'train')
blobs_multiple_test = Get_Next_Instance_HO_Neg(Trainval_GT, ctrl = 'test')

# blobs_multiple_train = blobs_multiple_train[:500]
# blobs_multiple_test = blobs_multiple_test[:100]

compute_cooccurence(blobs_multiple_train, blobs_multiple_test)

video_dir = "/home/rishabh/scene_graph/hoi_graph/dataset/All_subjects_images/"

print('Train no of batches', len(blobs_multiple_train))
print('Test no of batches', len(blobs_multiple_test))

def sigmoid_cross_entropy_loss(logits, labels, mask):
    logits, labels, mask = logits[0], labels[0], mask[0]
    # print(logits.shape, labels.shape, mask.shape)
    loss_1 = logits[logits>=0] - logits[logits>=0]*labels[logits>=0] + torch.log(1 + torch.exp(-logits[logits>=0]))
    loss_1 = loss_1*mask[logits>=0]
    loss_2 = - logits[logits<0]*labels[logits<0] + torch.log(1 + torch.exp(logits[logits<0]))
    loss_2 = loss_2*mask[logits<0]
    # print(loss_1.shape, loss_2.shape)
    loss = torch.mean(torch.cat((loss_1, loss_2), 0))
    return loss

prev_epoch_loss = 10000
for epoch in range(TRAIN_EPOCHS + 1): 
    timer = Timer()
    timer.tic()
    human_subactivity_acc = 0.
    object_subactivity_acc = 0.
    # combined_subactivity_acc = 0.
    
    log_train_file = open("log_train_file_"+mode+'_2', "a")
    log_test_file = open("log_test_file_"+mode+'_2', "a")
    # res_log_file = open("res_log", "a")
    # res_log_file.write("Epoch "+str(epoch)+"\n")

    final_data_len = len(blobs_multiple_train)
    loss_train, loss_train_1 = 0.0, 0.0

    batchNum = 0
    for blobs in blobs_multiple_train:
        # print('BatchNum', batchNum)
        st_time = time.time()
        batchNum += 1

        human_video_segment, object_video_segment = None, None
        if 'i3d' in mode:
            human_video_segment, object_video_segment = get_video_data(blobs)
        
        blobs['support'] = support
        blobs['features'] = features
        blobs['num_features_nonzero'] = features[1].shape
        blobs['iv_all'] = np.array(iv_all)

        H_spatial_feats = torch.Tensor(blobs['human_pose']).float().cuda()
        O_spatial_feats = torch.Tensor(blobs['obj_bboxes']).float().cuda()
        label_H = np.argmax(blobs['gt_class_H'])
        label_O = np.argmax(blobs['gt_class_O'])

        # print('Time to load data', time.time()-st_time)
        inputs = [human_video_segment, object_video_segment, H_spatial_feats, O_spatial_feats]
        subact_cls_scores, afford_cls_scores = model.forward(inputs,mode=mode) 

        # print(subact_cls_scores, label_H)
        # print(afford_cls_scores, label_O)

        human_loss_1 = criterion(subact_cls_scores, torch.Tensor([label_H]).long().cuda())
        object_loss_1 = criterion(afford_cls_scores, torch.Tensor([label_O]).long().cuda())
        subact_cls_scores = subact_cls_scores.cpu()
        afford_cls_scores = afford_cls_scores.cpu()

        # print('gt_labels', blobs['gt_class_H'], blobs['gt_class_O'])
        # print('mask', blobs['Mask_H'], blobs['Mask_O'])

        human_loss = sigmoid_cross_entropy_loss(subact_cls_scores, torch.Tensor(blobs['gt_class_H']), torch.Tensor(blobs['Mask_H']))
        object_loss = sigmoid_cross_entropy_loss(afford_cls_scores, torch.Tensor(blobs['gt_class_O']), torch.Tensor(blobs['Mask_O']))
        
        loss_1 = human_loss_1 + object_loss_1
        loss = human_loss + object_loss

        optimizer.zero_grad()
        loss.backward()
        # loss_1.backward()
        optimizer.step()
        
        del blobs

        loss_train += loss.item()
        loss_train_1 += loss_1.item()

        # print('Total time', time.time()-st_time, '\n')
        ####################
        H_pred  = np.argmax(subact_cls_scores.detach().numpy())
        O_pred  = np.argmax(afford_cls_scores.detach().numpy())

        if label_H == H_pred:
            human_subactivity_acc += 1.0

        if label_O == O_pred:
            object_subactivity_acc += 1.0
        

    human_subactivity_acc = (human_subactivity_acc / float(final_data_len)) * 100.00
    object_subactivity_acc = (object_subactivity_acc / float(final_data_len)) * 100.00

    timer.toc()
    if epoch % 1 == 0:           
        print('Epoch:%02d, loss: %.5f, loss_1: %.5f, human_acc: %.4f, object_acc: %.4f, lr: %f, time: %.3f s/iter' %  
              (epoch, loss_train/final_data_len, loss_train_1/final_data_len,  human_subactivity_acc, object_subactivity_acc, learning_rate, timer.average_time))
        log_train_file.write('Epoch:%02d, loss: %.5f, loss_1: %.5f, human_acc: %.4f, object_acc: %.4f, lr: %f, time: %.3f s/iter\n' %  
              (epoch, loss_train/final_data_len, loss_train_1/final_data_len, human_subactivity_acc, object_subactivity_acc, learning_rate, timer.average_time))
    # every epoch validation
    
    loss_val, loss_val_1 = 0.0, 0.0 
    test_human_subactivity_acc = 0.
    test_object_subactivity_acc = 0.

    timer.tic()  
    
    test_final_data_len = len(blobs_multiple_test)
    test_batchNum = 0
    for blobs in blobs_multiple_test:
        # print('Test BatchNum', test_batchNum)
        test_batchNum += 1

        human_video_segment, object_video_segment = None, None
        if 'i3d' in mode:
            human_video_segment, object_video_segment = get_video_data(blobs)

        blobs['support'] = support
        blobs['features'] = features
        blobs['num_features_nonzero'] = features[1].shape
        blobs['iv_all'] = np.array(iv_all)

        H_spatial_feats = torch.Tensor(blobs['human_pose']).float().cuda()
        O_spatial_feats = torch.Tensor(blobs['obj_bboxes']).float().cuda()
        test_label_H = np.argmax(blobs['gt_class_H'])
        test_label_O = np.argmax(blobs['gt_class_O'])

        inputs = [human_video_segment, object_video_segment, H_spatial_feats, O_spatial_feats]
        subact_cls_scores, afford_cls_scores = model.forward(inputs,mode=mode) 

        human_loss_1 = criterion(subact_cls_scores, torch.Tensor([test_label_H]).long().cuda())
        object_loss_1 = criterion(afford_cls_scores, torch.Tensor([test_label_O]).long().cuda())
        subact_cls_scores = subact_cls_scores.cpu()
        afford_cls_scores = afford_cls_scores.cpu()
        human_loss = sigmoid_cross_entropy_loss(subact_cls_scores, torch.Tensor(blobs['gt_class_H']), torch.Tensor(blobs['Mask_H']))
        object_loss = sigmoid_cross_entropy_loss(afford_cls_scores, torch.Tensor(blobs['gt_class_O']), torch.Tensor(blobs['Mask_O']))
        
        loss = human_loss + object_loss
        loss_val += loss.item()
        loss_1 = human_loss_1 + object_loss_1
        loss_val_1 += loss_1.item()

        ####################
        test_H_pred  = np.argmax(subact_cls_scores.detach().numpy())
        test_O_pred  = np.argmax(afford_cls_scores.detach().numpy())

        if test_label_H == test_H_pred:
            test_human_subactivity_acc += 1.0

        if test_label_O == test_O_pred:
            test_object_subactivity_acc += 1.0
        del blobs

    test_human_subactivity_acc = (test_human_subactivity_acc / float(test_final_data_len)) * 100.00
    test_object_subactivity_acc = (test_object_subactivity_acc / float(test_final_data_len)) * 100.00

    timer.toc()

    loss_val = loss_val/test_final_data_len
    loss_val_1 = loss_val_1/test_final_data_len
    print('Test time={:.3f} s/iter'.format(timer.average_time) + \
        ', eval_loss={:.5f}'.format(loss_val) + ', eval_loss_1={:.5f}'.format(loss_val_1) + \
        ', test_subact: %.4f'%(test_human_subactivity_acc) + \
        ', test_afford: %.4f\n'%(test_object_subactivity_acc)) 
    log_test_file.write('Test time={:.3f} s/iter'.format(timer.average_time) + \
        ', eval_loss={:.5f}'.format(loss_val) + ', eval_loss_1={:.5f}'.format(loss_val_1) + \
        ', test_subact: %.4f'%(test_human_subactivity_acc) + \
        ', test_afford: %.4f\n'%(test_object_subactivity_acc))
    log_train_file.close()
    log_test_file.close()

    if loss_train/final_data_len > prev_epoch_loss :
        scheduler.step()
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
        print('decrease lr to', learning_rate)
    prev_epoch_loss = loss_train/final_data_len

    if loss_val<loss_save[-1]:
        loss_save.append(loss_val)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'learning_rate': learning_rate,
            'gamma': gamma,
            }, True)
        # print(loss_save)
    else:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'learning_rate': learning_rate,
            'gamma': gamma,
            }, False)
