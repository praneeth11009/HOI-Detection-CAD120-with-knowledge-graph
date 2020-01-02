import os
import sys
root = '/home/nilay/hoi_graph'

import tensorflow as tf 
import numpy as np
import scipy.io as sio
import pickle
from collections import OrderedDict
import argparse
import ipdb
import cv2

sys.path.append(root + '/src3/gcn/')
from utils import *
from config_vcoco import cfg
import metadata_vcoco as metadata
from apply_prior import apply_prior
from vsrl_eval import VCOCOeval
sys.path.append(root + '/src3/gcn/test_files')

####################################
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print('added {} to pythonpath'.format(path))

# this_dir = osp.dirname(__file__)

# Add pycocotools to PYTHONPATH
coco_path = os.path.join('/home/nilay/v-coco', 'coco', 'PythonAPI')
add_path(coco_path)

from pycocotools.coco import COCO 
####################################

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
    im_file  = data_dir + 'v-coco/images/traintest2017/' + str(image_id).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape


parser = argparse.ArgumentParser(description='Test an iKNOW on VCOCO')
parser.add_argument('--savename', type=str, 
                    default='triple_gcn-09-29-06-48')
parser.add_argument('--model', dest='model',
                    help='Select model',
                    default='Triple_GCN', type=str)
parser.add_argument('--prior_flag', dest='prior_flag', 
                    help='whether use prior_flag',
                    default=3, type=int)
parser.add_argument('--object_thres', dest='object_thres',
                    help='Object threshold',
                    default=0.4, type=float)
parser.add_argument('--human_thres', dest='human_thres',
                    help='Human threshold',
                    default=0.8, type=float) 
parser.add_argument('--gpu', dest='gpu',
                    help='Specify GPU(s)',
                    default='2 4', type=str)
args = parser.parse_args()
    
if args.model == 'Triple_GCN':
    from iKNOW_VCOCO import Triple_GCN
    model_func = Triple_GCN
    modelname = 'Triple_GCN'.lower()
    
savename = args.savename
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
is_training = False

data_dir = root + '/data/'
output_dir = root + '/output/'
vcoco_dir = data_dir + 'v-coco'

dataset = 'glove_vcoco_vrd'
model_path = output_dir + '%s/%s/%s_latest.ckpt'%(dataset, savename, modelname)  # latest weight works for VCOCO
save_path = output_dir + 'vcoco/%s.pkl'%savename

meta = h5py.File(vcoco_dir + '/devkit/vcoco_meta.h5','r')
wordlist_file = data_dir + 'list/words_vcoco_vrd.json'
with open(wordlist_file) as fp:
    wordlist = json.load(fp)

features, support = load_graph(data_dir, dataset)
iv_all = []
for i in range(26):
    word = meta['meta/pre/idx2name/' + str(i)][...]
    iv_all.append(wordlist.index(str(word)[2:-1]))

Test_RCNN  = pkl.load( open( data_dir + 'Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl', "rb" ), encoding='latin1' )
Test_RCNN_Keypoints  = pkl.load( open( data_dir + 'Test_Faster_RCNN_R-50-PFN_2x_VCOCO_Keypoints_normalized.pkl', "rb" ), encoding='latin1' )
prior_mask = pkl.load( open( root + '/src3/gcn/test_files/prior_mask.pkl', 'rb'), encoding='latin1' )
Action_dic =json.load( open( data_dir + 'action_index.json'))
Action_dic_inv = {y:x for x,y in Action_dic.items()}
role_cats_all, obj_cats_all, instr_cats_all, role_names = relevant_vcoco_sets()
rel_verbs, _ = relevant_vcoco_verbs(wordlist, meta)
hoi_list = {0: 'surf_instr', 1: 'ski_instr', 2: 'cut_instr', 3: 'walk', 4: 'cut_obj', 5: 'ride_instr', 6: 'talk_on_phone_instr', 7: 'kick_obj', 8: 'work_on_computer_instr', 9: 'eat_obj', 10: 'sit_instr', 11: 'jump_instr', 12: 'lay_instr', 13: 'drink_instr', 14: 'carry_obj', 15: 'throw_obj', 16: 'eat_instr', 17: 'smile', 18: 'look_obj', 19: 'hit_instr', 20: 'hit_obj', 21: 'snowboard_instr', 22: 'run', 23: 'point_instr', 24: 'read_obj', 25: 'hold_obj', 26: 'skateboard_instr', 27: 'stand', 28: 'catch_obj'} # the same order 29 HOI GTs in iCAN

hoilist = [] # convert 26 to 29
for i, hoi in hoi_list.items():
    idx = [j for j in range(26) if str(meta['meta/pre/idx2name/' + str(j)][...])[2:-1] in hoi][0]
    hoilist.append(idx)
    

Testval_GT  = pkl.load( open( data_dir + 'Testval_GT_VCOCO_Keypoints_normalized.pkl', "rb" ), encoding='latin1' )
Trainval_GT = pkl.load( open( data_dir + 'Trainval_GT_VCOCO_Keypoints_normalized.pkl', "rb" ), encoding='latin1' )
coco_data_dir='/home/nilay/v-coco/coco'
dataType='trainval2017'
annFile2='{}/annotations2017/person_keypoints_{}.json'.format(coco_data_dir, dataType)
coco_kps = COCO(annFile2)

#####################################################
myimagedict = {}
for GT in Testval_GT:
    if GT[0] not in myimagedict:
        myimagedict[GT[0]] = []
        myimagedict[GT[0]].append((GT[2], GT[-1]))
    else:
        myimagedict[GT[0]].append((GT[2], GT[-1]))

for GT in Trainval_GT:
    if GT[0] not in myimagedict:
        myimagedict[GT[0]] = []
        myimagedict[GT[0]].append((GT[2], GT[-1]))
    else:
        myimagedict[GT[0]].append((GT[2], GT[-1]))

for image_id_ in Test_RCNN_Keypoints.keys():
    for GT in Test_RCNN_Keypoints[image_id_]:
       if GT[0] not in myimagedict:
        myimagedict[GT[0]] = []
        myimagedict[GT[0]].append((GT[2], GT[-1]))
    else:
        myimagedict[GT[0]].append((GT[2], GT[-1])) 

#####################################################

model = model_func(is_training=is_training)
saver = tf.train.Saver() 

with tf.Session() as sess:
    saver.restore(sess, model_path)
    print('Loaded network {:s}'.format(model_path))   
     
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    np.random.seed(cfg.RNG_SEED)
    cnt=0
    result = []
    
    for imid in Test_RCNN.keys(): 
        
        _t['im_detect'].tic()        
              
        im_orig, im_shape = get_blob(imid)

        blobs = {}
        blobs['H_num']       = 1
        blobs['support'] = support
        blobs['features'] = features
        blobs['num_features_nonzero'] = features[1].shape
        blobs['iv_all'] = np.array(iv_all)

        for Human_out in Test_RCNN_Keypoints[imid]: # detected humans

            if (np.max(Human_out[5]) > args.human_thres) and (Human_out[1] == 'Human'): 
                scoreh = func_object(Human_out[5])
                
                # Predict action using human appearance only
                blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)
                #probs_H  = model.test_image_H(sess, im_orig, blobs)
                #probs_H  = np.array(probs_H)

                ###########################

                # humanbox = np.array(Human_out[2])
                # mindist = None
                # minhumanpose = None
                # for humanboxes in myimagedict[imid]:
                #     if mindist == None:
                #         mindist = np.sum((humanbox  - np.array(humanboxes[0]))**2)
                #         minhumanpose = humanboxes[1]
                #     elif mindist > np.sum((humanbox  - np.array(humanboxes[0]))**2):
                #         mindist = np.sum((humanbox  - np.array(humanboxes[0]))**2)
                #         minhumanpose = humanboxes[1]
                #     else:
                #         continue

                minhumanpose = Human_out[-1]
                # import pdb
                # pdb.set_trace()
                # print(minhumanpose)
                num_joints_coord = len(minhumanpose)
                blobs['H_poses'] = minhumanpose.reshape(1, num_joints_coord)
                ###########################
                
                # save image information
                dic = {}
                dic['image_id']   = str(imid).zfill(6)
                dic['person_box'] = Human_out[2]

                # Predict action using human and object appearance 
                Score_obj     = np.empty((0, 4 + 29), dtype=np.float32) 

                for Object in Test_RCNN[imid]: # detected objects

                    if (np.max(Object[5]) > args.object_thres) and not (np.all(Object[2] == Human_out[2])): 
                    #if (np.max(scoreo) > args.object_thres) and (Object[1] == 'Object'): # no H-H
                        scoreo = func_object(Object[5])
                        
                        blobs['O_boxes'] = np.array([0,Object[2][0],Object[2][1],Object[2][2],Object[2][3]]).reshape(1,5)
                        
                        box_H = blobs['H_boxes'][0,1:].reshape(1,4)
                        box_O = blobs['O_boxes'][0,1:].reshape(1,4)
                        O_encode = bbox_transform(box_H, box_O)
                        H_encode = bbox_transform(box_O, box_H)
                        HO_encode = H_encode - O_encode  
                        blobs['H_boxes_enc']  = H_encode
                        blobs['O_boxes_enc']  = O_encode
                        blobs['HO_boxes_enc'] = HO_encode
                        
                        output_HO, output_O, output_H, sim = model.test_image_HO(sess, im_orig, blobs) 
                        probs = np.array(output_HO*output_O*output_H*softmax(sim)).reshape(1,26)
                        
                        probs_H = np.array(output_H)
                        
                        # convert to len=29
                        prediction_HO = [probs[0,idx] for idx in hoilist] 
                        prediction_H  = [probs_H[0,idx] for idx in hoilist]                        
                        
                        # filter 29 verb predictions based on object category
                        if args.prior_flag == 1:
                            prediction_HO  = apply_prior(Object[4], prediction_HO)
                        if args.prior_flag == 2:
                            prediction_HO  = prediction_HO * prior_mask[:,Object[4]].reshape(1,29) 
                        if args.prior_flag == 3:
                            prediction_HO  = apply_prior(Object[4], prediction_HO)
                            prediction_HO  = prediction_HO * prior_mask[:,Object[4]].reshape(1,29)
                        prediction_H = np.array(prediction_H).reshape(1,29)
                        
                        
                        This_Score_obj = np.concatenate((Object[2].reshape(1,4), prediction_HO * np.max(scoreo)), axis=1)   
                        Score_obj      = np.concatenate((Score_obj, This_Score_obj), axis=0)
                        
        
                if Score_obj.shape[0] == 0:
                    continue
                
                # Find out the object box associated with highest action score
                max_idx = np.argmax(Score_obj,0)[4:] 

                # agent mAP
                for i in range(29):
                    #'''
                    # walk, smile, run, stand
                    if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                        agent_name      = Action_dic_inv[i] + '_agent'
                        dic[agent_name] = np.max(scoreh) * prediction_H[0, i]
                        continue

                    # cut
                    if i == 2:
                        agent_name = 'cut_agent'
                        dic[agent_name] = np.max(scoreh) * max(Score_obj[max_idx[2]][4 + 2], Score_obj[max_idx[4]][4 + 4])
                        continue 
                    if i == 4:
                        continue   

                    # eat
                    if i == 9:
                        agent_name = 'eat_agent'
                        dic[agent_name] = np.max(scoreh) * max(Score_obj[max_idx[9]][4 + 9], Score_obj[max_idx[16]][4 + 16])
                        continue  
                    if i == 16:
                        continue

                    # hit
                    if i == 19:
                        agent_name = 'hit_agent'
                        dic[agent_name] = np.max(scoreh) * max(Score_obj[max_idx[19]][4 + 19], Score_obj[max_idx[20]][4 + 20])
                        continue  
                    if i == 20:
                        continue  

                    # These 2 classes need to save manually because there is '_' in action name
                    if i == 6:
                        agent_name = 'talk_on_phone_agent'  
                        dic[agent_name] = np.max(scoreh) * Score_obj[max_idx[i]][4 + i]
                        continue

                    if i == 8:
                        agent_name = 'work_on_computer_agent'  
                        dic[agent_name] = np.max(scoreh) * Score_obj[max_idx[i]][4 + i]
                        continue 

                    # all the rest
                    agent_name =  Action_dic_inv[i].split("_")[0] + '_agent'  
                    dic[agent_name] = np.max(scoreh) * Score_obj[max_idx[i]][4 + i]
                    #'''

                    '''
                    if i == 6:
                        agent_name = 'talk_on_phone_agent'  
                        dic[agent_name] = np.max(scoreh) * prediction_H[0][0][i]
                        continue

                    if i == 8:
                        agent_name = 'work_on_computer_agent'  
                        dic[agent_name] = np.max(scoreh) * prediction_H[0][0][i]
                        continue 

                    agent_name =  Action_dic_inv[i].split("_")[0] + '_agent'  
                    dic[agent_name] = np.max(scoreh) * prediction_H[0][0][i]
                    '''

                # role mAP
                for i in range(29):
                    # walk, smile, run, stand. Won't contribute to role mAP
                    if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                        dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(scoreh) * prediction_H[0, i]) 
                        continue

                    # Impossible to perform this action based on prior filters
                    if np.max(scoreh) * Score_obj[max_idx[i]][4 + i] == 0:
                        dic[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), np.max(scoreh) * Score_obj[max_idx[i]][4 + i])

                    # Action with >0 score
                    else:
                        dic[Action_dic_inv[i]] = np.append(Score_obj[max_idx[i]][:4], np.max(scoreh) * Score_obj[max_idx[i]][4 + i])

                result.append(dic)

            
        _t['im_detect'].toc()
            
        print('im_detect: {:d}/{:d} {:.3f}s'.format(cnt + 1, 4946, _t['im_detect'].average_time))
        cnt+=1
       
    with open(save_path, 'wb') as f:
        pkl.dump(result, f, pickle.HIGHEST_PROTOCOL) 

# Evaluation 
vcocoeval = VCOCOeval(data_dir + 'vcocoeval/vcoco/vcoco_test.json', data_dir + 'vcocoeval/instances_vcoco_all_2014.json', data_dir + 'vcocoeval/splits/vcoco_test.ids')  
vcocoeval._do_eval(save_path, ovr_thresh=0.5) 
print(save_path)
