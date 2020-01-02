import pickle as pkl
import numpy as np
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print('added {} to pythonpath'.format(path))


def normalize_human_pose(pose):
	pose = (np.array(pose).reshape(17, 3))[:, :2]
	pose = pose - np.mean(pose, 0)
	norm = np.sqrt(np.mean(np.sum(pose**2, 1)))

	if norm == 0:
		# print(pose)
		# print(1)
		return pose.reshape(34)
	
	normalize_pose = pose / np.sqrt(np.mean(np.sum(pose**2, 1)))
	return normalize_pose.reshape(34)

# this_dir = osp.dirname(__file__)

# Add pycocotools to PYTHONPATH
coco_path = osp.join('/home/nilay/v-coco', 'coco', 'PythonAPI')
add_path(coco_path)

from pycocotools.coco import COCO

data_dir = '/home/nilay/hoi_graph/data/'
coco_data_dir='/home/nilay/v-coco/coco'
dataType='trainval2017'
# input_pkl_file = 'Trainval_GT_VCOCO'
input_pkl_file = 'Trainval_Neg_VCOCO'
annFile2='{}/annotations2017/person_keypoints_{}.json'.format(coco_data_dir, dataType)

# coco=COCO(annFile1)
coco_kps = COCO(annFile2)

Trainval_neg_GT = pkl.load( open( data_dir + input_pkl_file + '.pkl', "rb" ), encoding='latin1')

num_zero_pose = 0
num_poses = 0
for image_id in list(Trainval_neg_GT.keys()):
	for neg_example in Trainval_neg_GT[image_id]:
		flag = 0
		agent_bbox = np.array(neg_example[2])
		annotations_coco = coco_kps.loadAnns(coco_kps.getAnnIds(imgIds=image_id))
		for a in annotations_coco:
			bbox = np.array(a['bbox'])
			if bbox[0] == agent_bbox[0] and bbox[1] == agent_bbox[1]:
				pose_ = normalize_human_pose(a['keypoints'])
				if np.sum(pose_) == 0:
					num_zero_pose += 1
				num_poses += 1
				neg_example.append(pose_)
				flag += 1

		if flag==0:
			print(neg_example)
			print(agent_bbox)
			print(np.array([a['bbox'] for a in annotations_coco]))
			
		assert(flag==1)

pkl.dump(Trainval_neg_GT, open( data_dir + input_pkl_file + '_Keypoints_normalized.pkl', "wb"), protocol=pkl.HIGHEST_PROTOCOL)
print("Dumping into ", data_dir + input_pkl_file + '_Keypoints_normalized.pkl')
print("Num Zero poses : ", num_zero_pose)
print("Total Poses : ", num_poses)