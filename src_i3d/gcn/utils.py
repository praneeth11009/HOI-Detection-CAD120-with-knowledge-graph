import numpy as np
import pickle as pkl
import json
import h5py
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import os
import tensorflow as tf
import time
import datetime
import pdb

import metadata_vcoco

vcocoroot = '/home/nilay/v-coco/'
sys.path.append(vcocoroot)
sys.path.append(vcocoroot+'coco/PythonAPI/')
import vsrl_utils as vu

import logging
from colorlog import ColoredFormatter

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
#    datefmt='%H:%M:%S.%f',
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'white,bold',
        'INFOV':    'cyan,bold',
        'WARNING':  'yellow',
        'ERROR':    'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('rn')
log.setLevel(logging.DEBUG)
log.handlers = []       # No duplicated handlers
log.propagate = False   # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')
def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)

logging.Logger.infov = _infov


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sample_mask_sigmoid(idx, h, w):
    """Create mask."""
    mask = np.zeros((h, w))
    matrix_one = np.ones((h, w))
    mask[idx, :] = matrix_one[idx, :]
    return np.array(mask, dtype=np.bool)


def load_data_vis_multi(dataset_str, use_trainval, feat_suffix, label_suffix='ally_multi'):
    """Load data."""
    names = [feat_suffix, label_suffix, 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.NELL.{}".format(dataset_str, names[i]), 'rb') as f:
            print("{}/ind.NELL.{}".format(dataset_str, names[i]))
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    allx, ally, graph = tuple(objects)
    train_test_mask = []
    with open("{}/ind.NELL.index".format(dataset_str), 'rb') as f:
        train_test_mask = pkl.load(f)

    features = allx  # .tolil()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    labels = np.array(ally)

    idx_test = []
    idx_train = []
    idx_trainval = []

    if use_trainval == True:
        for i in range(len(train_test_mask)):

            if train_test_mask[i] == 0:
                idx_train.append(i)
            if train_test_mask[i] == 1:
                idx_test.append(i)

            if train_test_mask[i] >= 0:
                idx_trainval.append(i)
    else:
        for i in range(len(train_test_mask)):

            if train_test_mask[i] >= 0:
                idx_train.append(i)
            if train_test_mask[i] == 1:
                idx_test.append(i)

            if train_test_mask[i] >= 0:
                idx_trainval.append(i)

    idx_val = idx_test

    train_mask = sample_mask_sigmoid(idx_train, labels.shape[0], labels.shape[1]) 
    val_mask = sample_mask_sigmoid(idx_val, labels.shape[0], labels.shape[1])
    trainval_mask = sample_mask_sigmoid(idx_trainval, labels.shape[0], labels.shape[1])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_trainval = np.zeros(labels.shape)

    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_trainval[trainval_mask] = labels[trainval_mask]

    return adj, features, y_train, y_val, y_trainval, train_mask, val_mask, trainval_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def preprocess_features_dense(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def preprocess_features_dense2(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    div_mat = sp.diags(rowsum)

    return features, div_mat


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def create_config_proto():
    """Reset tf default config proto"""
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 0
    config.gpu_options.force_gpu_compatible = True
    # config.operation_timeout_in_ms=8000
    config.log_device_placement = False
    return config

def load_graph(data_dir, dataset):
    names = ['allx_dense', 'graph']
    dataset_str = os.path.join(data_dir, dataset)
    objects = []
    for i in range(len(names)):
        with open("{}/ind.NELL.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    allx, graph = tuple(objects)
    # print(objects)
    features = allx  
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) 
    
    features, div_mat = preprocess_features_dense2(features)
    support = [preprocess_adj(adj)] # nonzero values and their coordinates [row, col] in the sparse adj matrix
    num_supports = 1 # len(support)
    return features, support
    
# def get_iv_hico(rels, meta, wordlist):
#     iv = []
#     for rel in rels: # sample
#         assert len(rel)==117
#         tmp = []
#         for i in range(117):
#             if rel[i]==1:            
#                 word = meta['meta/pre/idx2name/' + str(i)][...]
#                 tmp.append(wordlist.index(str(word)))
#             else:
#                 tmp.append(1000) 
#         iv.append(tmp)
#     return iv

def get_iv_vcoco(rels, meta, wordlist):
    iv = []
    for rel in rels: # sample
        assert len(rel)==26
        tmp = []
        for i in range(26):
            if rel[i]==1:            
                word = meta['meta/pre/idx2name/' + str(i)][...]
                tmp.append(wordlist.index(str(word)))
            else:
                tmp.append(1000) 
        iv.append(tmp)
    return iv

def next_batch(batchid, batch_size, c, v, l, y, vh, vo):
    """ Return a batch of data. When dataset end is reached, start over.
    """
    nsamples = len(y)
    if batchid == nsamples:
        batchid = 0
    batchids = np.arange(nsamples)[batchid:min(batchid + batch_size, nsamples)]
    y_batch = np.array(y)[batchids]
    c_batch = np.array(c)[batchids]
    v_batch = np.array(v)[batchids]
    l_batch = np.array(l)[batchids]
    vh_batch = np.array(vh)[batchids]
    vo_batch = np.array(vo)[batchids]
    batchid = min(batchid + batch_size, nsamples)
    
    return c_batch, v_batch, l_batch, y_batch, vh_batch, vo_batch, batchids, batchid

def construct_feeddict_gcn(support, features, y, feats, idvnames, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    feed_dict.update({placeholders['features']: np.array(features)})
    feed_dict.update({placeholders['visual']: np.array(feats)})
    feed_dict.update({placeholders['idvnames']: np.array(idvnames)})
    if list(y):
        feed_dict.update({placeholders['labels']: np.array(y)})
    return feed_dict

def construct_feeddict(c, v, l, vh, vo, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['classeme']: np.array(c)})
    feed_dict.update({placeholders['visual']: np.array(v)})
    feed_dict.update({placeholders['loc']: np.array(l)})
    feed_dict.update({placeholders['vis_h']: np.array(vh)})
    feed_dict.update({placeholders['vis_o']: np.array(vo)})
    return feed_dict

def evaluate(sess, model, support, features, c, v, vh, vo, l, y, iv_all, placeholders, batch_size, is_train, modelname, epoch, val_writer, merge, count_val):
    t_val = time.time()
    nsamples = len(y)
    s = np.random.permutation(np.arange(nsamples))
    y = np.array(y)[s]
    c = np.array(c)[s]  
    v = np.array(v)[s]
    vh = np.array(vh)[s]
    vo = np.array(vo)[s]
    l = np.array(l)[s]
    
    loss = 0.0
    batchid = 0
    batch_num = int(len(y)/batch_size)
    for batch in range(batch_num): 
        count_val += 1
        c_batch, v_batch, l_batch, y_batch, vh_batch, vo_batch, batchids, batchid = next_batch(batchid, batch_size, c, v, l, y, vh, vo)
        
        feed_dict = construct_feeddict(c_batch, v_batch, l_batch, vh_batch, vo_batch, placeholders)
        if is_train:
            feed_dict.update({placeholders['labels']: np.array(y_batch)})
        if modelname in ['triple_gcn', 'triple_gcn_early', 'triple_wogcn']:
            feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
            feed_dict.update({placeholders['features']: np.array(features)})
            feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
            feed_dict.update({placeholders['iv_all']: np.array(iv_all)})
            
        summary, total_loss = sess.run([merge, model.losses['total_loss']], feed_dict=feed_dict) 
        val_writer.add_summary(summary, count_val) 
        loss = loss + total_loss

    loss = loss/batch_num
    return loss, (time.time() - t_val), count_val

def test(sess, model, support, features, c, v, vh, vo, l, y, iv_all, placeholders, batch_size, is_train, modelname):
    t_val = time.time()
    nsamples = len(y)
    s = np.random.permutation(np.arange(nsamples))
    y = np.array(y)[s]
    c = np.array(c)[s]  
    v = np.array(v)[s]
    vh = np.array(vh)[s]
    vo = np.array(vo)[s]
    l = np.array(l)[s]
    
    loss = 0.0
    batchid = 0
    batch_num = int(len(y)/batch_size)
    for batch in range(batch_num): 
        c_batch, v_batch, l_batch, y_batch, vh_batch, vo_batch, batchids, batchid = next_batch(batchid, batch_size, c, v, l, y, vh, vo)
        
        feed_dict = construct_feeddict(c_batch, v_batch, l_batch, vh_batch, vo_batch, placeholders)
        if is_train:
            feed_dict.update({placeholders['labels']: np.array(y_batch)})
        if modelname in ['triple_gcn', 'triple_gcn_early', 'triple_wogcn']:
            feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
            feed_dict.update({placeholders['features']: np.array(features)})
            feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
            feed_dict.update({placeholders['iv_all']: np.array(iv_all)})
        
        total_loss = sess.run(model.losses['total_loss'], feed_dict=feed_dict)   # no weight update  
        loss = loss + total_loss
        
    loss = loss/batch_num
    return loss, (time.time() - t_val)

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K))
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def bbox_transform(ex_rois, gt_rois):  #y1 x1 y2 x2
    '''
    # sub_box_encoded = bbox_transform(np.array([locations[ob]]),np.array([locations[s]]))[0]
    '''
    # anchor sizes
    ex_heights = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_widths = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    # ex_time = ex_rois[:, 5] - ex_rois[:, 4] + 1.0
    ex_ctr_y = ex_rois[:, 0] + 0.5 * ex_heights
    ex_ctr_x = ex_rois[:, 1] + 0.5 * ex_widths
    # ex_ctr_z = ex_rois[:, 4] + 0.5 * ex_time
    
    
    # exact BB coordinates
    gt_heights = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_widths = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    # gt_time = gt_rois[:, 5] - gt_rois[:, 4] + 1.0
    gt_ctr_y = gt_rois[:, 0] + 0.5 * gt_heights
    gt_ctr_x = gt_rois[:, 1] + 0.5 * gt_widths
    # gt_ctr_z = gt_rois[:, 4] + 0.5 * gt_time
    
    # rpn_bbox layer output BB regression
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    # targets_dz = (gt_ctr_z - ex_ctr_z) / ex_time
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    # targets_dt = np.log(gt_time / ex_time)

    targets = np.vstack(
        (targets_dy, targets_dx, targets_dh, targets_dw)).transpose()
    return targets

# def relevant_hico_sets(meta, list_action):
#     objs_all = []
#     for i in range(117):
#         predicate = meta['meta/pre/idx2name/'+str(i)][...]
#         predicate = str(predicate)
#         obj_all = []
#         for ind in range(len(list_action)):
#             if [list_action[ind]][0].vname==predicate:
#                 obj_all.append([list_action[ind]][0].nname) 
#         objs_all.append(obj_all)
#     return objs_all

def relevant_vcoco_sets():
    coco = vu.load_coco(vcocoroot + 'data/')
    vsrl_data = vu.load_vcoco('vcoco_train', vcocoroot + 'data/') 

    obj_cats_all = []
    instr_cats_all = []
    role_cats_all = []
    role_names = []
    actions = []
    for i in range(len(vsrl_data)): # 26 actions, same order as in meta.h5 and action.mat
        action = vsrl_data[i]['action_name']
        role_ids = vsrl_data[i]['role_object_id']
        role_name = vsrl_data[i]['role_name']
        
        role_names.append(role_name)
        actions.append(action) 
        
        role_cats = []
        obj_cats = []
        instr_cats = []
        if 'obj' in role_name:
            objcol = role_name.index('obj')
            obj_ids = role_ids[:,objcol]
            obj_ids = obj_ids.tolist()
            obj_ids = [x for x in obj_ids if x != 0]
            for obj_id in obj_ids:
                obj_cat = vu.coco_obj_id_to_obj_class(int(obj_id), coco)
                if obj_cat not in obj_cats:
                    obj_cats.append(obj_cat)
                    role_cats.append(obj_cat)
        if 'instr' in role_name:
            instrcol = role_name.index('instr')
            instr_ids = role_ids[:,instrcol]
            instr_ids = instr_ids.tolist()
            instr_ids = [x for x in instr_ids if x != 0]
            for instr_id in instr_ids:
                instr_cat = vu.coco_obj_id_to_obj_class(int(instr_id), coco)
                if instr_cat not in instr_cats:
                    instr_cats.append(instr_cat)
                    role_cats.append(instr_cat)

        obj_cats_all.append(obj_cats) 
        instr_cats_all.append(instr_cats)  
        role_cats_all.append(role_cats)
    return role_cats_all, obj_cats_all, instr_cats_all, role_names

def relevant_vcoco_verbs(wordlist, meta):
    role_cats_all, _, _, _ = relevant_vcoco_sets()
    actnames = []
    iv_rel = []
    for i in range(80):
        obj_name = meta['meta/cls/idx2name/' + str(i)][...]
        aids = [a for a in range(26) if str(obj_name) in role_cats_all[a]]
        tmp = []
        ivtmp = []
        for a in aids:
            act = meta['meta/pre/idx2name/' + str(a)][...] # has underlines
            tmp.append(str(act))
            ivtmp.append(wordlist.index(str(act)))
        actnames.append(tmp)
        iv_rel.append(ivtmp)
        
    return actnames, iv_rel

def read_roidb(roidb_path):
    roidb_file = np.load(roidb_path)
    key = roidb_file.keys()[0]
    roidb_temp = roidb_file[key]
    roidb = roidb_temp[()]
    return roidb

import time

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff