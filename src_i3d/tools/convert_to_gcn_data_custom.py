# --------------------------------------------------------
# Based on code of zero-shot-gcn
# --------------------------------------------------------
import argparse
import json
import numpy as np
import pickle as pkl
from scipy import sparse
import os
import scipy.io as sio
import h5py
import sys

from tensorflow.python import pywrap_tensorflow

# from prepare_list import prepare_graph

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
root = '/home/rishabh/scene_graph/hoi_graph/v-coco-master/' # vsrl toolkit dir
add_path(root)
add_path(root+'coco/PythonAPI/')
import vsrl_utils as vu

data_dir = '/home/rishabh/scene_graph/hoi_graph/data/'
# dataset = 'hico_vrd'
dataset = 'cad120'

def convert_to_gcn_data(wv_file):
    save_dir = os.path.join(data_dir, '%s_%s' % (args.wv, dataset))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Converting input')
    convert_input(wv_file, save_dir)
    print('Converting graph')
    convert_graph(save_dir)
    print('Prepared data to %s' % save_dir)

def convert_input(wv_file, save_dir):
    with open(wv_file, 'rb') as fp:
        feats = pkl.load(fp)
    feats = feats.tolist()
    sparse_feats = sparse.csr_matrix(feats)
    dense_feats = np.array(feats)

    sparse_file = os.path.join(save_dir, 'ind.NELL.allx')
    dense_file = os.path.join(save_dir, 'ind.NELL.allx_dense')
    
    with open(sparse_file, 'wb') as fp:
        pkl.dump(sparse_feats, fp)
    with open(dense_file, 'wb') as fp:
        pkl.dump(dense_feats, fp)

    print('Save feat in shape to', sparse_file, dense_file, 'with shape', dense_feats.shape)
    return
    
def convert_label(save_dir):  
    '''save labels'''
    # Load detections and annotations
    hico_dir = data_dir + 'hico_20160224_det'
    anno_file = os.path.join(hico_dir, 'devkit/anno_bbox.mat')
    anno = sio.loadmat(anno_file, struct_as_record=False, squeeze_me=True) 
    list_action = anno['list_action'] 
    meta_file = os.path.join(hico_dir, 'devkit/hico_det_meta.h5')
    meta = h5py.File(meta_file, 'r')
    
    
    idx = np.random.choice(len(anno['bbox_train'])) 
    bbox = anno['bbox_train'][idx] # 'filename','hoi','size'
    imid = bbox.filename[:-4]
    hois = bbox.hoi # 'id', 'bboxhuman', 'bboxobject', 'connection', 'invis'
    
    humans = []
    objects = []
    relations = []
    onames = []
    if not type(hois)==np.ndarray: 
        hois = [hois]
    for hoi in hois:
        relation = np.zeros((117))
        if hoi.invis: continue      
        hid = hoi.id-1 # 0-599 
        predicate = list_action[hid].vname
        oname = list_action[hid].nname
        aid = meta['meta/pre/name2idx/' + predicate][...] # 0-116
        aid = aid.astype(int)
        relation[aid] = 1

        if not type(hoi.bboxhuman)==np.ndarray: 
            bboxhumans = [hoi.bboxhuman]
        else:
            bboxhumans = hoi.bboxhuman
        if not type(hoi.bboxobject)==np.ndarray: 
            bboxobjs = [hoi.bboxobject]
        else:
            bboxobjs = hoi.bboxobject
        conn = hoi.connection
        if conn.ndim==1:
            conn = conn.reshape(1,2)
        for i in xrange(conn.shape[0]): # sub amounts
            human = bboxhumans[conn[i,0]-1] 
            sub = [human.x1, human.y1, human.x2, human.y2]
            ob = bboxobjs[conn[i,1]-1] 
            obj = [ob.x1, ob.y1, ob.x2, ob.y2]

            idx = [i for i,v in enumerate(humans) if v==sub and objects[i]==obj]
            if idx: # the same ho pair
                assert len(idx)==1
                relations[idx[0]][aid] = 1
            else: # new ho pair
                humans.append(sub)
                objects.append(obj)
                relations.append(relation)
                onames.append(oname)
    assert len(humans)==len(objects)==len(relations)
    print('relations dim', np.array(relations).shape) 

    label_file = os.path.join(save_dir, 'ind.NELL.ally_multi') 
    with open(label_file, 'wb') as fp:
        pkl.dump(relations, fp)
    return

def convert_graph(save_dir):
    graph_file = os.path.join(data_dir, '%s_graph.pkl'%dataset) 
    if not os.path.exists(graph_file):
        # prepare_graph()
        print('File', graph_file, 'does not exist')
    save_file = os.path.join(save_dir, 'ind.NELL.graph') 
    if os.path.exists(save_file):
        cmd = 'rm  %s' % save_file
        os.system(cmd)
    cmd = 'ln -s %s %s' % (graph_file, save_file)
    os.system(cmd)
    return


def parse_arg():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hop', type=str, default='all',
                        help='choice of unseen set: 2,3,all')
    parser.add_argument('--fc', type=str, default='res50',
                        help='choice: [inception,res50]')
    parser.add_argument('--wv', type=str, default='glove',
                        help='word embedding type: [glove, google, fasttext]')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()

    # print('Args wv', args.wv)

    if args.wv == 'glove':
        wv_file = os.path.join(data_dir, 'word_embedding_model', 'glove_%s.pkl'%dataset)
    elif args.wv == 'google':
        wv_file = os.path.join(data_dir, 'word_embedding_model', 'google_word2vec_wordnet.pkl')
    elif args.wv == 'fasttext':
        wv_file = os.path.join(data_dir, 'word_embedding_model', 'fasttext_word2vec_wordnet.pkl')
    else:
        raise NotImplementedError

    convert_to_gcn_data(wv_file)
