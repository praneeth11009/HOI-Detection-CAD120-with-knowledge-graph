import os
import torch
import torch.nn as nn

from config_vcoco import cfg
from metrics import *
from layers import GraphConvolution

import numpy as np
import ipdb
import sys
import time

sys.path.append('/home/rishabh/scene_graph/hoi_graph/src_py/gcn/pytorch-i3d/')
from pytorch_i3d import *

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)

class Triple_GCN(nn.Module):
    def __init__(self, is_training, mode='i3d'):
        super(Triple_GCN, self).__init__()
        self.H_feat_size_i3d = 512
        self.O_feat_size_i3d = 512
        self.H_feat_size_spatial = 500
        self.O_feat_size_spatial = 200

        if mode == 'i3d_spatial':
            self.H_feat_size = 512 + 500
            self.O_feat_size = 512 + 200
        elif mode == 'i3d':
            self.H_feat_size = 512
            self.O_feat_size = 512
        elif mode == 'spatial':
            self.H_feat_size = 500
            self.O_feat_size = 200

        self.num_classes_human = 10
        self.num_classes_object = 12

        self.train = is_training
        self.i3d_model = InceptionI3d(final_endpoint='Mixed_5c')
        self.i3d_model.float().cuda()

        self.process_i3d_human = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=1, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1]),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=[3, 3, 3], stride=[3, 3, 1], padding=[0, 0, 1]),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=[0, 0, 0])
        )

        self.process_i3d_object = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=1, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1]),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=[3, 3, 3], stride=[3, 3, 1], padding=[0, 0, 1]),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=[0, 0, 0])
        )

        self.visual_feature_human = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.1),
            nn.Linear(512, self.H_feat_size_i3d),
            nn.Dropout(0.1)
        )
        self.visual_feature_object = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.1),
            nn.Linear(512, self.O_feat_size_i3d),
            nn.Dropout(0.1)
        )

        # Processing for spatial input ## Diff Exp
        self.H_spatial = nn.Sequential(
            nn.Linear(900, 500),
            nn.Dropout(0.1),
            nn.Linear(500, 500),
            nn.Dropout(0.1),
            nn.Linear(500, self.H_feat_size_spatial),
            nn.Dropout(0.1),
        )
        self.O_spatial = nn.Sequential(
            nn.Linear(200, 200),
            nn.Dropout(0.1),
            nn.Linear(200, 200),
            nn.Dropout(0.1),
            nn.Linear(200, self.O_feat_size_spatial),
            nn.Dropout(0.1),
        )

        self.H_subact_cls = nn.Sequential(
            nn.Linear(self.H_feat_size + self.O_feat_size, 512),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_classes_human),
            nn.Softmax()
        )
        self.O_afford_cls = nn.Sequential(
            nn.Linear(self.H_feat_size + self.O_feat_size, 512),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_classes_object),
            nn.Softmax()
        )

        self.i3d_model.apply(weights_init)
        self.process_i3d_human.apply(weights_init)
        self.process_i3d_object.apply(weights_init)
        self.visual_feature_human.apply(weights_init)
        self.visual_feature_object.apply(weights_init)
        self.H_spatial.apply(weights_init)
        self.O_spatial.apply(weights_init)
        self.H_subact_cls.apply(weights_init)
        self.O_afford_cls.apply(weights_init)
    
    def forward(self, inputs, mode='i3d'):
        human_video_segment = inputs[0]
        object_video_segment = inputs[1]

        H_spatial_input = inputs[2].reshape([1,900])
        O_spatial_input = inputs[3].reshape([1,200])

        batch_size = H_spatial_input.shape[0]

        H_fc, O_fc = None, None
        if mode == 'i3d': 
            i3d_human_feats = self.i3d_model.extract_features(human_video_segment).permute(0, 2, 3, 4, 1)
            i3d_object_feats = self.i3d_model.extract_features(object_video_segment).permute(0, 2, 3, 4, 1)

            i3d_human_feats = self.process_i3d_human(i3d_human_feats)
            i3d_object_feats = self.process_i3d_object(i3d_object_feats)

            i3d_human_feats = i3d_human_feats.reshape([batch_size, 512])
            i3d_object_feats = i3d_object_feats.reshape([batch_size, 512])

            H_fc = self.visual_feature_human(i3d_human_feats)
            O_fc = self.visual_feature_object(i3d_object_feats)
        elif mode == 'spatial':
            H_fc = self.H_spatial(H_spatial_input)
            O_fc = self.O_spatial(O_spatial_input)
        elif mode == 'i3d_spatial':
            i3d_human_feats = self.i3d_model.extract_features(human_video_segment).permute(0, 2, 3, 4, 1)
            i3d_object_feats = self.i3d_model.extract_features(object_video_segment).permute(0, 2, 3, 4, 1)

            i3d_human_feats = self.process_i3d_human(i3d_human_feats)
            i3d_object_feats = self.process_i3d_object(i3d_object_feats)

            i3d_human_feats = i3d_human_feats.reshape([batch_size, 512])
            i3d_object_feats = i3d_object_feats.reshape([batch_size, 512])

            H_fc_i3d = self.visual_feature_human(i3d_human_feats)
            O_fc_i3d = self.visual_feature_object(i3d_object_feats)

            H_fc_spatial = self.H_spatial(H_spatial_input)
            O_fc_spatial = self.O_spatial(O_spatial_input)

            H_fc = torch.cat((H_fc_i3d, H_fc_spatial), 1)
            O_fc = torch.cat((O_fc_i3d, O_fc_spatial), 1)

        concat = torch.cat((H_fc, O_fc), 1)
        subact_cls_scores = self.H_subact_cls(concat)
        afford_cls_scores = self.O_afford_cls(concat)

        return subact_cls_scores, afford_cls_scores 




   
    
     

    
