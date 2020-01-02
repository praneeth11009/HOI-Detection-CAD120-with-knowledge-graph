import pickle as pkl
import numpy as np
import os.path as osp
import sys
import scipy.ndimage.filters as fi
import matplotlib.pyplot as plt
import skimage.io as io
import time

# base_video_dir = "/home/nilay/hoi_graph/src_3D/gcn/CAD120/All_subjects_images/Subject1_rgbd_images"
base_video_dir = "/home/nilay/hoi_graph/src_3D/gcn/CAD120/All_subjects_images"
# base_annontations_path = "/home/nilay/hoi_graph/src_3D/gcn/CAD120/All_subjects_annotations/Subject1_annotations"
base_annontations_path = "/home/nilay/hoi_graph/src_3D/gcn/CAD120/All_subjects_annotations"
human_affordance_map = {}
human_affordance_map['reaching'] = 0
human_affordance_map['moving'] = 1
human_affordance_map['pouring'] = 2
human_affordance_map['eating'] = 3
human_affordance_map['drinking'] = 4
human_affordance_map['opening'] = 5
human_affordance_map['placing'] = 6
human_affordance_map['closing'] = 7
human_affordance_map['scrubbing'] = 8
human_affordance_map['null'] = 9
human_affordance_map['cleaning'] = 10

object_affordance_map = {}
object_affordance_map['reachable'] = 0
object_affordance_map['movable'] = 1
object_affordance_map['pourable'] = 2
object_affordance_map['pourto'] = 3
object_affordance_map['containable'] = 4
object_affordance_map['drinkable'] = 5
object_affordance_map['openable'] = 6
object_affordance_map['placeable'] = 7
object_affordance_map['closeable'] = 8
object_affordance_map['scrubbable'] = 9
object_affordance_map['scrubber'] = 10
object_affordance_map['stationary'] = 11
object_affordance_map['cleanable'] = 12

high_level_activity_map = {}
high_level_activity_map['making_cereal'] = 0
high_level_activity_map['taking_medicine'] = 1
high_level_activity_map['stacking_objects'] = 2
high_level_activity_map['unstacking_objects'] = 3
high_level_activity_map['microwaving_food'] = 4
high_level_activity_map['picking_objects'] = 5
high_level_activity_map['cleaning_objects'] = 6
high_level_activity_map['taking_food'] = 7
high_level_activity_map['arranging_objects'] = 8
high_level_activity_map['having_meal'] = 9

# high_level_activity_list = ['making_cereal', 'taking_medicine', 'stacking_objects', 'unstacking_objects',\
#  'microwaving_food', 'picking_objects', 'cleaning_objects', 'taking_food', 'arranging_objects', 'having_meal']

high_level_activity_list = ['making_cereal', 'taking_medicine', 'stacking_objects', 'unstacking_objects',\
 'microwaving_food', 'picking_objects', 'cleaning_objects', 'taking_food', 'arranging_objects']


# Subject_list = ['Subject1', 'Subject3', 'Subject4', 'Subject5']
# Subject_list = ['Subject1', 'Subject3', 'Subject4']
Subject_list = ['Subject5']

even_more_complete_GT = {}
all_video_ids = []

for Subject in Subject_list:
	annontations_path = base_annontations_path + "/" + Subject + "_" + "annotations"
	complete_GT = {}
	complete_video_ids = []
	for action in high_level_activity_list:
		activity_annotations = annontations_path + "/" + action + "/" + "activityLabel.txt"
		subactivity_annotations = annontations_path + "/" + action + "/" + "labeling.txt"

		# Construct GT
		subact = open(subactivity_annotations, 'r')
		act = open(activity_annotations, 'r')

		subact_lines = subact.readlines()
		act_lines = act.readlines()

		Video_Ids = []
		GT = {}

		for line in subact_lines:
			words = line.strip().split(',')
			video_id = words[0]
			if video_id not in Video_Ids:
				Video_Ids.append(video_id)
				GT[str(video_id)] = {}
				GT[str(video_id)]['high_level_activity'] = None
				GT[str(video_id)]['subject'] = None
				GT[str(video_id)]['object_type'] = None
				GT[str(video_id)]['segments'] = []

			segment = {}
			start_frame = int(words[1])
			end_frame = int(words[2])
			segment['start'] = start_frame
			segment['end'] = end_frame
			segment['human_affordance'] = human_affordance_map[str(words[3])]
			segment['object_affordance'] = object_affordance_map [str(words[4])]
			segment['human_bboxes'] = np.zeros([end_frame - start_frame + 1, 4])
			segment['object_bboxes'] = np.zeros([end_frame - start_frame + 1, 4])
			GT[str(video_id)]['segments'].append(segment)


		for line in act_lines:
			words = line.strip().split(',')
			video_id = words[0]
			GT[str(video_id)]['high_level_activity'] = high_level_activity_map[str(action)]
			GT[str(video_id)]['subject'] = str(words[2])
			GT[str(video_id)]['Subject_name'] = Subject
			GT[str(video_id)]['object_type'] = str(words[3])

		joint_idx = np.zeros([1 + (10+4)*11 + (4*4)]).astype('int')
		for join_num in range(11):
			joint_idx[1 + (10 + 4)*join_num + 10 + 1 - 1] = 1
			joint_idx[1 + (10 + 4)*join_num + 10 + 2 - 1] = 1

		for join_num in range(4):
			joint_idx[1 + (10 + 4)*11 + (4*join_num) + 1 - 1] = 1
			joint_idx[1 + (10 + 4)*11 + (4*join_num) + 2 - 1] = 1	

		joint_idx = joint_idx == 1


		#Add humans and objects
		human_files = []
		object_files = []

		curr_start_frame = None
		curr_end_frame = None
		curr_seg = 0

		# import pdb
		# pdb.set_trace()

		for video_id in Video_Ids:
			human_file = open(annontations_path + "/" + action + "/" + str(video_id) + ".txt", 'r')
			object_file = open(annontations_path + "/" + action + "/" + str(video_id) + "_obj1.txt", 'r')
			human_file_lines = np.array(human_file.readlines())
			object_file_lines = np.array(object_file.readlines())
			num_images = human_file_lines.shape[0] - 1
			# print(num_images)
			# print(object_file_lines.shape[0])
			if num_images != object_file_lines.shape[0]:
				print(action)
				print(video_id)
				assert(num_images == object_file_lines.shape[0])

			human_coord = np.stack([np.array(line.strip().split(','))[:-1] for line in human_file_lines][:-1], axis=0)
			object_coord = np.stack([np.array(line.strip().split(',')) for line in object_file_lines][:-1], axis=0)
			num_segments = len(GT[str(video_id)]['segments'])

			for curr_seg in range(num_segments):
				curr_start_frame = GT[str(video_id)]['segments'][curr_seg]['start']
				curr_end_frame = GT[str(video_id)]['segments'][curr_seg]['end']

				human_joint_bboxes = (human_coord[curr_start_frame-1:curr_end_frame][:, joint_idx].reshape([curr_end_frame - curr_start_frame + 1, 15, 2]).astype('float'))
				human_bboxes = np.zeros([curr_end_frame - curr_start_frame + 1, 4]).astype('float')
				x_max = np.max(human_joint_bboxes[:, :, 0], axis=1).astype('float') + 1000.00
				x_min = np.min(human_joint_bboxes[:, :, 0], axis=1).astype('float') + 1000.00
				y_max = np.max(human_joint_bboxes[:, :, 1], axis=1).astype('float') + 1000.00
				y_min = np.min(human_joint_bboxes[:, :, 1], axis=1).astype('float') + 1000.00

				human_bboxes[:, 0] = np.max(np.stack((y_min - 20, np.zeros(y_min.shape[0]).astype('float')), axis=1), axis=1)
				human_bboxes[:, 1] = np.max(np.stack((x_min - 20, np.zeros(x_min.shape[0]).astype('float')), axis=1), axis=1)
				human_bboxes[:, 2] = np.min(np.stack((y_max + 20, np.zeros(y_max.shape[0]).astype('float') + 2000.00), axis=1), axis=1)
				human_bboxes[:, 3] = np.min(np.stack((x_max + 20, np.zeros(x_max.shape[0]).astype('float') + 2000.00), axis=1), axis=1)

				human_bboxes = human_bboxes / 2000.00
				
				object_bboxes = (object_coord[curr_start_frame-1:curr_end_frame][:, 2:6].astype('float'))
				object_bboxes[:, 0] /= 640.00
				object_bboxes[:, 2] /= 640.00
				object_bboxes[:, 1] /= 480.00
				object_bboxes[:, 3] /= 480.00

				GT[str(video_id)]['segments'][curr_seg]['human_bboxes'] = human_bboxes
				GT[str(video_id)]['segments'][curr_seg]['object_bboxes'] = object_bboxes
				human_joint_bboxes_norm = (human_joint_bboxes + 1000.000) / 2000.000

				# Normalize human_joint_bboxes_norm
				mean_human_joint_boxes = np.mean(human_joint_bboxes_norm, axis=1).reshape(human_joint_bboxes_norm.shape[0],\
				 1, human_joint_bboxes_norm.shape[2]).repeat(human_joint_bboxes_norm.shape[1], 1)

				human_joint_bboxes_norm -= mean_human_joint_boxes

				variance_human_joint_boxes = np.mean(human_joint_bboxes_norm**2, axis=1).reshape(human_joint_bboxes_norm.shape[0], \
					1, human_joint_bboxes_norm.shape[2]).repeat(human_joint_bboxes_norm.shape[1], 1)

				human_joint_bboxes_norm /= np.sqrt(variance_human_joint_boxes)
				####################################

				GT[str(video_id)]['segments'][curr_seg]['human_joint_bboxes_norm'] = human_joint_bboxes_norm

		complete_video_ids += Video_Ids
		complete_GT.update(GT)

	assert(len(complete_video_ids) == len(set(complete_video_ids)))
	all_video_ids += complete_video_ids
	even_more_complete_GT.update(complete_GT)

assert(len(all_video_ids) == len(set(all_video_ids)))
pkl.dump(even_more_complete_GT, open( './CAD120_GT_all_subjects_testing_new_wo_having_meal_with_norm_pose.pkl', "wb"), protocol=pkl.HIGHEST_PROTOCOL)
print("Dumping into  ./CAD120_GT_all_subjects_testing_new_wo_having_meal_with_norm_pose.pkl")

