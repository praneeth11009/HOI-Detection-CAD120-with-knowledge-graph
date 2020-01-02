import os
import sys
root = '/home/hoi_graph'
sys.path.append(root + '/src/gcn/')
sys.path.append(root+'/src/gcn/test_files')
import tensorflow as tf
import numpy as np
import scipy.io as sio
import pickle as pkl
import ipdb
import cv2
import glob
import time
from collections import OrderedDict
import argparse

from utils import *
import metadata_hico
from config_hico import cfg

def func(x):
    # Calculate rank of similarity
    a = {}
    rank = 0
    for num in sorted(x[0]):
        if num not in a:
            a[num] = rank
            rank   = rank+1
    ranks = np.array([a[i] for i in x[0]])
    return ranks/float(np.max(ranks))

def func_object(prob): 
    return (np.exp(8*prob)-1)/(np.exp(8)-1)

def softmax(x):
    # convert cosine similarity (-1,1) to (0,1)
    return 1/(1+np.exp(-x))

def get_blob(image_id):
    im_file  = data_dir + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape



parser = argparse.ArgumentParser(description='Test an iKNOW on HICO')
parser.add_argument('--savename', type=str, 
                    default='triple_gcn-02-05-18-22')
parser.add_argument('--model', dest='model',
                    help='Select model',
                    default='Triple_GCN', type=str)
parser.add_argument('--object_thres', dest='object_thres',
                    help='Object threshold',
                    default=0.8, type=float)
parser.add_argument('--human_thres', dest='human_thres',
                    help='Human threshold',
                    default=0.6, type=float)
parser.add_argument('--gpu', dest='gpu',
                    help='Specify GPU(s)',
                    default='0', type=str) # "0,1"
args = parser.parse_args()
  
if args.model == 'Triple_GCN':
    from iKNOW_HICO import Triple_GCN
    model_func = Triple_GCN
    modelname = 'Triple_GCN'.lower()
    
savename = args.savename
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
is_training = False

data_dir = root + '/data/'
output_dir = root + '/output/'
hico_dir = data_dir + 'hico_20160224_det'

dataset = 'glove_hico_vrd'
model_path = output_dir + '%s/%s/%s_best.ckpt'%(dataset, savename, modelname) # best weight works for HICO
det_file = output_dir + 'hico/%s.mat'%savename

Test_RCNN = pkl.load( open( data_dir + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl', "rb" ) )
meta = h5py.File(os.path.join(hico_dir, 'devkit/hico_det_meta.h5'),'r')  
anno_file = os.path.join(hico_dir, 'devkit/anno_bbox.mat')
anno = sio.loadmat(anno_file, struct_as_record=False, squeeze_me=True)
annobbox = anno['bbox_test']
list_action = anno['list_action'] 
 wordlist_file = data_dir + 'list/words_hico_vrd.json'
with open(wordlist_file) as fp:
    wordlist = json.load(fp)

features, support = load_graph(data_dir,  dataset)
iv_all = []
for i in range(117):
    word = meta['meta/pre/idx2name/' + str(i)][...]
    iv_all.append(wordlist.index(str(word)))
    
model = model_func(is_training=is_training)
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_path)
    print('Loaded network {:s}'.format(model_path))
   
    _t = {'im_detect' : Timer(), 'misc' : Timer()}        
    np.random.seed(cfg.RNG_SEED)
    count = 0
    
    objs_all = relevant_hico_sets(meta, list_action)
    all_boxes = [[[] for j in range(9658)] for i in range(600)]

    for i in range(9658):

        _t['im_detect'].tic()        

        image_id = int(annobbox[i].filename[-9:-4]) # the same order as evaluation images

        im_orig, im_shape = get_blob(image_id)

        blobs = {}
        blobs['H_num'] = 1
        blobs['support'] = support
        blobs['features'] = features
        blobs['num_features_nonzero'] = features[1].shape
        blobs['iv_all'] = np.array(iv_all)

        for Human_out in Test_RCNN[image_id]:
            if (np.max(Human_out[5]) > args.human_thres) and (Human_out[1] == 'Human'): 

                blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)

                for Object in Test_RCNN[image_id]:
                    if (np.max(Object[5]) > args.object_thres) and not (np.all(Object[2] == Human_out[2])): 

                        blobs['O_boxes'] = np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5)
                                              
                        box_H = blobs['H_boxes'][0,1:].reshape(1,4)
                        box_O = blobs['O_boxes'][0,1:].reshape(1,4)
                        O_encode = bbox_transform(box_H, box_O)
                        H_encode = bbox_transform(box_O, box_H)
                        HO_encode = H_encode - O_encode 
                        blobs['H_boxes_enc']  = H_encode
                        blobs['O_boxes_enc']  = O_encode
                        blobs['HO_boxes_enc'] = HO_encode
                        
                        prob_HO, prob_H, prob_O, sim = model.test_image_HO(sess, im_orig, blobs) 
                        probs = np.array(prob_HO*prob_H*prob_O*softmax(sim)).reshape(1,117)
                        
                        obj_pred = metadata_hico.coco_classes[Object[4]]
                        rel_v = metadata_hico.obj_actions[Object[4]] # 1-person, len=80
                        for a in rel_v:
                            action = metadata_hico.action_classes[a] # has underlines
                            hoi_ind = [idx for idx in range(600) if [list_action[idx]][0].nname==obj_pred 
                                       and [list_action[idx]][0].vname==str(action)][0] 
                            
                            score_h = func_object(Human_out[5])
                            score_o = func_object(Object[5])
                            score_hoi = np.array([score_h*score_o*probs[0, a]]).reshape(1,1)
                            boxes_score = np.concatenate((box_H, box_O, score_hoi), axis=1)
                            all_boxes[hoi_ind][count].append(boxes_score) 

        
        _t['im_detect'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 9658, _t['im_detect'].average_time))
        count += 1 
    

    sio.savemat(det_file, {'all_boxes' : all_boxes})    
    print('Saved to: ' + det_file)

