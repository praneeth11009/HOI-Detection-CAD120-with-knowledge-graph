# ------------------------------
# Based on code from Tensorflow iCAN
# ------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import ops

from config_vcoco import cfg
from metrics import *
from layers import GraphConvolution

import numpy as np
import ipdb

flags = tf.app.flags
FLAGS = flags.FLAGS

def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor

def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }
    with arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
        weights_initializer = slim.variance_scaling_initializer(),
        biases_regularizer  = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), 
        biases_initializer  = tf.constant_initializer(0.0),
        trainable           = is_training,
        activation_fn       = tf.nn.relu,
        normalizer_fn       = slim.batch_norm,
        normalizer_params   = batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

        
class Triple_GCN():
    def __init__(self, is_training):
        self.name = self.__class__.__name__.lower()
        self.vars = {}      
        self.layers = []
        self.activations = []  
        self.visualize = {}
        self.intermediate = {}
        self.predictions = {}
        self.losses = {}
        self.score_summaries = {}
        self.event_summaries = {}
        
        
        self.image       = tf.placeholder(tf.float32, shape=[1, None, None, 3], name = 'image')
        self.H_boxes     = tf.placeholder(tf.float32, shape=[None , 5], name = 'H_boxes')
        self.O_boxes     = tf.placeholder(tf.float32, shape=[None , 5], name = 'O_boxes')
        self.H_boxes_enc = tf.placeholder(tf.float32, shape=[None , 4], name = 'H_boxes_enc')
        self.O_boxes_enc = tf.placeholder(tf.float32, shape=[None , 4], name = 'O_boxes_enc')
        self.HO_boxes_enc= tf.placeholder(tf.float32, shape=[None , 4], name = 'HO_boxes_enc')        
        self.gt_class_HO = tf.placeholder(tf.float32, shape=[None, 117], name = 'gt_class_HO')
        self.H_num       = tf.placeholder(tf.int32) # pos
        self.ivs         = tf.placeholder(tf.int32, shape=[117], name = 'idx_GT_verbs')
        self.inputs      = tf.placeholder(tf.float32, shape=[None, 300], name = 'embedding')         
        self.support     = [tf.sparse_placeholder(tf.float32) for _ in range(1)]
        self.num_nonzero = tf.placeholder(tf.int32)
        self.in_dim      = 300
        self.hidden_dim  = 512
        self.out_dim     = 512
        self.num_classes = 117
        self.scope       = 'resnet_v1_50'
        self.stride      = [16, ]
        self.train       = is_training 
        
        self.now_lr      = None
        self.optimizer   = None
        self.opt_op      = None        
        if tf.__version__ == '1.1.0':
            self.blocks     = [resnet_utils.Block('block1', resnet_v1.bottleneck,[(256,   64, 1)] * 2 + [(256,   64, 2)]),
                               resnet_utils.Block('block2', resnet_v1.bottleneck,[(512,  128, 1)] * 3 + [(512,  128, 2)]),
                               resnet_utils.Block('block3', resnet_v1.bottleneck,[(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
                               resnet_utils.Block('block4', resnet_v1.bottleneck,[(2048, 512, 1)] * 3),
                               resnet_utils.Block('block5', resnet_v1.bottleneck,[(2048, 512, 1)] * 3)]
        else:
            from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
            self.blocks = [resnet_v1_block('block1', base_depth=64,  num_units=3, stride=2),
                           resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                           resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
                           resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
                           resnet_v1_block('block5', base_depth=512, num_units=3, stride=1)]
            
        self.build_all()

    def build_base(self):
        with tf.variable_scope(self.scope, self.scope):
            net = resnet_utils.conv2d_same(self.image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net

    def image_to_head(self):
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net    = self.build_base()
            net, _ = resnet_v1.resnet_v1(net,
                                         self.blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                         global_pool=False,
                                         include_root_block=False,
                                         scope=self.scope)
        with slim.arg_scope(resnet_arg_scope(is_training=self.train)):
            head, _ = resnet_v1.resnet_v1(net,
                                          self.blocks[cfg.RESNET.FIXED_BLOCKS:-2],
                                          global_pool=False,
                                          include_root_block=False,
                                          scope=self.scope)
        return head
    
    # Get pooled appearance feature from head
    def crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:

            batch_ids    = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            bottom_shape = tf.shape(bottom)
            height       = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.stride[0])
            width        = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

            bboxes        = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            if cfg.RESNET.MAX_POOL:
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
                crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
            else:
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE], name="crops")
        return crops

    def res5(self, pool5_H, pool5_O, name):
        with slim.arg_scope(resnet_arg_scope(is_training=self.train)):

            fc7_H, _ = resnet_v1.resnet_v1(pool5_H,
                                           self.blocks[-2:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc7_H = tf.reduce_mean(fc7_H, axis=[1, 2]) # TensorShape([Dimension(None), Dimension(2048)])


            fc7_O, _ = resnet_v1.resnet_v1(pool5_O,
                                       self.blocks[-1:], # note
                                       global_pool=False,
                                       include_root_block=False,
                                       reuse=False,
                                       scope=self.scope) # TensorShape([Dimension(None), Dimension(2048)])

            fc7_O = tf.reduce_mean(fc7_O, axis=[1, 2])
        
        return fc7_H, fc7_O

    
    def visual_feature(self, fc7_H, fc7_O, HO_box_enc, H_box_enc, O_box_enc, name): 
        with tf.variable_scope(name) as scope:
                           
            # HO 
            vis_fc1 = slim.fully_connected(fc7_H - fc7_O, 512, scope='vis_fc1')
            vis_fc1 = slim.dropout(vis_fc1, keep_prob=0.5, is_training=self.train, scope='vis_dropout1')
            
            loc_fc1 = slim.fully_connected(HO_box_enc, 20, scope='loc_fc1')
            loc_fc2 = slim.fully_connected(loc_fc1, 10, scope='loc_fc2')

            concat_HO = tf.concat([0.3*vis_fc1, loc_fc2], axis=1) 
            
            HO_fc2 = slim.fully_connected(concat_HO, 512, scope='HO_fc') 
            HO_fc2 = slim.dropout(HO_fc2, keep_prob=0.5, is_training=self.train, scope='HO_dropout')
                            
            # H 
            vis_H_fc1 = slim.fully_connected(fc7_H, 512, scope='vis_H_fc1')
            vis_H_fc1 = slim.dropout(vis_H_fc1, keep_prob=0.5, is_training=self.train, scope='vis_H_dropout1')
            
            loc_H_fc1 = slim.fully_connected(H_box_enc, 20, scope='loc_H_fc1')
            loc_H_fc2 = slim.fully_connected(loc_H_fc1, 10, scope='loc_H_fc2')
            
            concat_H = tf.concat([0.3*vis_H_fc1, loc_H_fc2], axis=1) 
            
            H_fc2 = slim.fully_connected(concat_H, 512, scope='H_fc2')
            H_fc2 = slim.dropout(H_fc2, keep_prob=0.5, is_training=self.train, scope='H_dropout2')
                
            # O
            vis_O_fc1 = slim.fully_connected(fc7_O, 512, scope='vis_O_fc1')
            vis_O_fc1 = slim.dropout(vis_O_fc1, keep_prob=0.5, is_training=self.train, scope='vis_O_dropout1')
            
            loc_O_fc1 = slim.fully_connected(O_box_enc, 20, scope='loc_O_fc1')
            loc_O_fc2 = slim.fully_connected(loc_O_fc1, 10, scope='loc_O_fc2')
            
            concat_O = tf.concat([0.3*vis_O_fc1, loc_O_fc2], axis=1)
            
            O_fc2 = slim.fully_connected(concat_O, 512, scope='O_fc')
            O_fc2 = slim.dropout(O_fc2, keep_prob=0.5, is_training=self.train, scope='O_dropout')
                
            return HO_fc2, H_fc2, O_fc2
    
    def region_classification(self, HO_fc2, H_fc2, O_fc2, initializer, name):
        with tf.variable_scope(name) as scope:  
                                  
            cls_score_HO  = slim.fully_connected(HO_fc2, self.num_classes, weights_initializer=initializer, 
                                                trainable=self.train, activation_fn=None, scope='cls_score_HO')
            cls_prob_HO   = tf.nn.sigmoid(cls_score_HO, name='cls_prob_HO')
            
            
            cls_score_H  = slim.fully_connected(H_fc2, self.num_classes, weights_initializer=initializer, 
                                                trainable=self.train, activation_fn=None, scope='cls_score_H')
            cls_prob_H   = tf.nn.sigmoid(cls_score_H, name='cls_prob_H') 
            
            
            cls_score_O  = slim.fully_connected(O_fc2, self.num_classes, weights_initializer=initializer, 
                                                trainable=self.train, activation_fn=None, scope='cls_score_O')
            cls_prob_O   = tf.nn.sigmoid(cls_score_O, name='cls_prob_O') 
            
            
            self.predictions["cls_score_HO"]= cls_score_HO
            self.predictions["cls_prob_HO"] = cls_prob_HO
            self.predictions["cls_score_H"] = cls_score_H
            self.predictions["cls_prob_H"]  = cls_prob_H
            self.predictions["cls_score_O"] = cls_score_O
            self.predictions["cls_prob_O"]  = cls_prob_O

    
    def build_GCN(self, inputs, in_dim, hidden_dim, out_dim): 
        placeholders = {}
        placeholders['support'] = self.support
        placeholders['num_features_nonzero'] = self.num_nonzero
        
        layer1 = GraphConvolution(input_dim=in_dim, output_dim=hidden_dim,
                                  placeholders=placeholders, act=lambda x: tf.maximum(x, 0.2 * x),
                                  dropout=False, sparse_inputs=False, name='gcn_layer1')
        out1 = layer1(inputs)

        layer2 = GraphConvolution(input_dim=hidden_dim, output_dim=out_dim,
                                  placeholders=placeholders, act=lambda x: tf.maximum(x, 0.2 * x),
                                  dropout=True, name='gcn_layer2')
        out2 = layer2(out1)
        self.emb_g = out2
                
        return out2
   
    def joint_embedding(self, gcn, HO_fc2, name):
        with tf.variable_scope(name) as scope:
            
            HO_norm = tf.nn.l2_normalize(HO_fc2, dim=1, name='HO_norm') 
            graph_norm = tf.nn.l2_normalize(gcn, dim=1, name='graph_emb_norm')             
            
            batchsize = tf.shape(HO_norm)[0]
       
            phis_im = tf_repeat(tf.expand_dims(HO_norm, axis=1), [1,self.num_classes,1])
            self.predictions['phi_im'] = phis_im       
            
            phi_graph = tf.gather(graph_norm, self.ivs) 
            phis_g = tf.tile(tf.expand_dims(phi_graph,0), [batchsize,1,1]) 
            self.predictions['phi_GCN'] = phis_g
            
            return phis_im, phis_g

    
    def add_loss(self):
        with tf.variable_scope('LOSS') as scope:
          
            cls_score_H  = self.predictions["cls_score_H"]
            cls_score_O  = self.predictions["cls_score_O"]
            cls_score_HO  = self.predictions["cls_score_HO"]
            phis_im = self.predictions['phi_im']
            phis_g  = self.predictions['phi_GCN']
            
            label_HO  = self.gt_class_HO 

            H_cross_entropy  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_HO[:self.H_num,:], logits = cls_score_H[:self.H_num,:]))
            O_cross_entropy  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_HO[:self.H_num,:], logits = cls_score_O[:self.H_num,:]))
            HO_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_HO, logits = cls_score_HO))
            
            sim_loss = CosineEmbeddingLoss(0.1)(phis_im, phis_g, label_HO) 
            
            loss = FLAGS.sim_weight * sim_loss + HO_cross_entropy + H_cross_entropy + O_cross_entropy
            
            self.losses['H_cross_entropy'] = H_cross_entropy
            self.losses['O_cross_entropy'] = O_cross_entropy
            self.losses['HO_cross_entropy'] = HO_cross_entropy
            
            self.losses['total_loss'] = loss
            self.event_summaries.update(self.losses)

        return loss 
    
    def build_all(self):            
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        head    = self.image_to_head()
        pool5_H = self.crop_pool_layer(head, self.H_boxes, 'Crop_H')
        pool5_O = self.crop_pool_layer(head, self.O_boxes, 'Crop_O')
        
        fc7_H, fc7_O = self.res5(pool5_H, pool5_O, 'res5')               
        HO_fc, H_fc, O_fc = self.visual_feature(fc7_H, fc7_O, self.HO_boxes_enc, 
                                                self.H_boxes_enc, self.O_boxes_enc, 'fc_feature') 
        self.region_classification(HO_fc, H_fc, O_fc, initializer, 'H_O_classification')
        
        GCN = self.build_GCN(self.inputs, self.in_dim, self.hidden_dim, self.out_dim)       
        phis_im, phis_g = self.joint_embedding(GCN, HO_fc, 'joint_embedding')
        

        if self.train: 
            
            loss = self.add_loss() 
            
            tf.summary.scalar("loss/cross_entropy", loss)
            
            self.vars= {var.name: var for var in tf.trainable_variables()}
            
            global_step = tf.Variable(0, trainable=False)
            self.now_lr  = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, 
                                                      cfg.TRAIN.STEPSIZE, cfg.TRAIN.GAMMA, staircase=True)
            #self.opt_op = tf.train.AdamOptimizer(learning_rate=self.now_lr).minimize(loss)
            self.optimizer = tf.train.MomentumOptimizer(self.now_lr, cfg.TRAIN.MOMENTUM)
            grads_and_vars = self.optimizer.compute_gradients(loss, tf.trainable_variables())
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars]            
            self.opt_op = self.optimizer.apply_gradients(capped_gvs, global_step=global_step) # global_step will count and update
        else:
            
            self.vars= {var.name: var for var in tf.model_variables(scope='H_O_classification')}       
            
            self.sim = cosine_similarity(phis_im, phis_g)

            
            
    def save_best(self, sess=None, savepath=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver() 
        save_path = savepath + "/%s_best.ckpt" % self.name
        saver.save(sess, save_path)
        print("Best model saved in file: %s" % save_path)       

    def load_best(self, sess=None, savepath=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver()
        save_path = savepath + "/%s_best.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Best model restored from file: %s" % save_path)
    
    def resume(self, sess=None, savepath=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(savepath))
        print("Resume training")        
     
    def train_step(self, sess, blobs):
        feed_dict = {self.image: blobs['image'], self.H_boxes: blobs['H_boxes'],
                     self.O_boxes: blobs['O_boxes'], self.gt_class_HO: blobs['gt_class_HO'], 
                     self.H_num: blobs['H_num'], self.H_boxes_enc: blobs['H_aug_enc'],
                     self.O_boxes_enc: blobs['O_aug_enc'], self.HO_boxes_enc: blobs['HO_aug_enc'],
                     self.inputs: blobs['features'], self.num_nonzero: blobs['num_features_nonzero'], 
                     self.ivs: blobs['iv_all']}
        feed_dict.update({self.support[i]: blobs['support'][i] for i in range(len(blobs['support']))})
        
        _, total_loss, now_lr = sess.run([self.opt_op, 
                                                     self.losses['total_loss'], 
                                                     self.now_lr], 
                                                     feed_dict=feed_dict)
        
        return total_loss, now_lr

    
    def train_step_with_summary(self, sess, blobs, merge):
        feed_dict = {self.image: blobs['image'], self.H_boxes: blobs['H_boxes'],
                     self.O_boxes: blobs['O_boxes'], self.gt_class_HO: blobs['gt_class_HO'], 
                     self.H_num: blobs['H_num'], self.H_boxes_enc: blobs['H_aug_enc'],
                     self.O_boxes_enc: blobs['O_aug_enc'], self.HO_boxes_enc: blobs['HO_aug_enc'],
                     self.inputs: blobs['features'], self.num_nonzero: blobs['num_features_nonzero'], 
                     self.ivs: blobs['iv_all']}
        feed_dict.update({self.support[i]: blobs['support'][i] for i in range(len(blobs['support']))})
       
        summary, _, total_loss, now_lr = sess.run([merge, 
                                                             self.opt_op, 
                                                             self.losses['total_loss'], 
                                                             self.now_lr], 
                                                             feed_dict=feed_dict)

        return summary, total_loss, now_lr
    
    
    def test_step(self, sess, blobs):
        # For trainval, no opt_op
        feed_dict = {self.image: blobs['image'], self.H_boxes: blobs['H_boxes'],
                     self.O_boxes: blobs['O_boxes'], self.gt_class_HO: blobs['gt_class_HO'], 
                     self.H_num: blobs['H_num'], self.H_boxes_enc: blobs['H_aug_enc'],
                     self.O_boxes_enc: blobs['O_aug_enc'], self.HO_boxes_enc: blobs['HO_aug_enc'],
                     self.inputs: blobs['features'], self.num_nonzero: blobs['num_features_nonzero'], 
                     self.ivs: blobs['iv_all']}
        feed_dict.update({self.support[i]: blobs['support'][i] for i in range(len(blobs['support']))})
        
        total_loss = sess.run([self.losses['total_loss']], feed_dict=feed_dict)

        return total_loss
    
    
    def test_step_with_summary(self, sess, blobs, merge):
        # For trainval, no opt_op
        feed_dict = {self.image: blobs['image'], self.H_boxes: blobs['H_boxes'],
                     self.O_boxes: blobs['O_boxes'], self.gt_class_HO: blobs['gt_class_HO'], 
                     self.H_num: blobs['H_num'], self.H_boxes_enc: blobs['H_aug_enc'],
                     self.O_boxes_enc: blobs['O_aug_enc'], self.HO_boxes_enc: blobs['HO_aug_enc'],
                     self.inputs: blobs['features'], self.num_nonzero: blobs['num_features_nonzero'], 
                     self.ivs: blobs['iv_all']}
        feed_dict.update({self.support[i]: blobs['support'][i] for i in range(len(blobs['support']))})
        
        summary, total_loss = sess.run([merge, self.losses['total_loss']], 
                                                  feed_dict=feed_dict)

        return summary, total_loss
    
    
    def test_image_HO(self, sess, image, blobs):
        feed_dict = {self.image: image, self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'], 
                     self.H_num: blobs['H_num'], self.H_boxes_enc:blobs['H_boxes_enc'],
                     self.O_boxes_enc:blobs['O_boxes_enc'], self.HO_boxes_enc: blobs['HO_boxes_enc'], 
                     self.inputs: blobs['features'], self.num_nonzero: blobs['num_features_nonzero'], 
                     self.ivs: blobs['iv_all']}
        feed_dict.update({self.support[i]: blobs['support'][i] for i in range(len(blobs['support']))})
            
        prob_HO, prob_H, prob_O, sim = sess.run([self.predictions["cls_prob_HO"], self.predictions["cls_prob_H"], self.predictions["cls_prob_O"], self.sim], feed_dict=feed_dict)

        return prob_HO, prob_H, prob_O, sim
    
    
    
    