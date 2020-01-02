import pickle as pkl
import numpy as np
import os.path as osp
import sys
import scipy.ndimage.filters as fi
import matplotlib.pyplot as plt
import skimage.io as io
import time

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
	return normalize_pose
	# return normalize_pose.reshape(34)

# def gkern(kernlen=21, nsig=3):
# 	"""Returns a 2D Gaussian kernel."""
# 	x = np.linspace(-nsig, nsig, kernlen+1)
# 	kern1d = np.diff(st.norm.cdf(x))
# 	kern2d = np.outer(kern1d, kern1d)
# 	return kern2d/kern2d.sum()

def gkern2(kernlen=21, nsig=3):
	"""Returns a 2D Gaussian kernel array."""
	# create nxn zeros
	inp = np.zeros((kernlen, kernlen))
	# set element at the middle to one, a dirac delta
	inp[kernlen//2, kernlen//2] = 1
	# gaussian-smooth the dirac, resulting in a gaussian filter mask
	return fi.gaussian_filter(inp, nsig)

def gkern3(l=5, sig=1.):
	"""
	creates gaussian kernel with side length l and a sigma of sig
	"""
	ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
	xx, yy = np.meshgrid(ax, ax)
	kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
	# return kernel
	return kernel / np.sum(kernel)

def Get_Pose_Heatmap(pose, sigma=1.0, normalize=True, mean=1.0):
	# import pdb
	# pdb.set_trace()
	heatmap = np.zeros([64, 64]).astype('float64')
	for x in pose:
		center_x = 32 + int(x[0]*8)
		center_y = 32 + int(x[1]*8)
		kernel = gkern3((6*int(sigma))+1, sigma)
		if normalize:
			kernel = kernel / np.sum(kernel)
		else:
			kernel = kernel*mean
		print(center_x, " ", center_y)  
		heatmap[center_x-(3*int(sigma)):center_x+(3*int(sigma))+1, center_y-(3*int(sigma)):center_y+(3*int(sigma))+1] += kernel
	return heatmap


def Get_Pose_Heatmap2(pose, image, sigma=1.0, normalize=True, mean=1.0):
	import pdb
	pdb.set_trace()
	heatmap = np.zeros([image.shape[0], image.shape[1]]).astype('float64')
	for x in pose:
		center_x = int(x[0])
		center_y = int(x[1])
		kernel = gkern3((6*int(sigma))+1, sigma)
		if normalize:
			kernel = kernel / np.sum(kernel)
		else:
			kernel = kernel*mean
		heatmap[center_x-(3*int(sigma)):center_x+(3*int(sigma))+1, center_y-(3*int(sigma)):center_y+(3*int(sigma))+1] = kernel
	return heatmap

def Get_posemap(pose, human, map_size=64, sigma=1):
	heatmap = np.zeros([len(pose), map_size, map_size]).astype('float64')
	pose = (np.array(pose).reshape(17, 3))
	mask = np.zeros([17]).astype('float64')
	min_x = np.min(pose[:, 0])
	min_y = np.min(pose[:, 1])
	max_x = np.max(pose[:, 0])
	max_y = np.max(pose[:, 1])
	width = max_x - min_x
	height = max_y - min_y
	index = 0
	
	for x in pose:
		if x[2] != 0:
			xcoord = ((x[0] - min_x)*64) / width
			ycoord = ((x[1] - min_y)*64) / height
			center_x = int(xcoord)
			center_y = int(ycoord)
			# print(center_x, " ", center_y, " ", x[2])
			
			left_x = max(center_x-(3*int(sigma)), 0)
			right_x = min(center_x+(3*int(sigma))+1, 64)
			left_y = max(center_y-(3*int(sigma)), 0)
			right_y = min(center_y+(3*int(sigma))+1, 64)
			
			kernel = gkern3((6*int(sigma))+1, sigma)[(3*int(sigma))-(center_x - left_x):(3*int(sigma))+(right_x - center_x), (3*int(sigma))-(center_y - left_y):(3*int(sigma))+(right_y - center_y)]
			heatmap[index, left_x:right_x, left_y:right_y] += kernel
			
			mask[index] = 1.0
		else:
			mask[index] = 0.0

	return heatmap, mask

# this_dir = osp.dirname(__file__)

# Add pycocotools to PYTHONPATH
coco_path = osp.join('/home/nilay/v-coco', 'coco', 'PythonAPI')
add_path(coco_path)

from pycocotools.coco import COCO

data_dir = '/home/nilay/hoi_graph/data/'
coco_data_dir='/home/nilay/v-coco/coco'
dataType='trainval2017'
# input_pkl_file = 'Trainval_GT_VCOCO'
input_pkl_file = 'Testval_GT_VCOCO'
# input_pkl_file = 'Test_Faster_RCNN_R-50-PFN_2x_VCOCO'
annFile2='{}/annotations2017/person_keypoints_{}.json'.format(coco_data_dir, dataType)

# coco=COCO(annFile1)
coco_kps = COCO(annFile2)

Trainval_GT = pkl.load( open( data_dir + input_pkl_file + '.pkl', "rb" ), encoding='latin1')

zero_poses_count = 0
total_poses = 0

for GT in Trainval_GT:	
	Image_id = GT[0]
	Human = GT[2]
	annsids = coco_kps.getAnnIds(imgIds=Image_id)
	annotations_coco = coco_kps.loadAnns(annsids)
	flag = 0
	for a in annotations_coco:
		bbox = np.array(a['bbox'])
		if bbox[0] == Human[0] and bbox[1] == Human[1]:
			total_poses += 1
			pose = np.array(a['keypoints'])
			# normalized_pose_ = normalize_human_pose(a['keypoints'])
			# if np.sum(normalized_pose_) == 0:
			# 	zero_poses_count += 1
			# 	continue
			# pose_heat_map = Get_Pose_Heatmap(normalized_pose_)
			# pose_heat_map, pose_mask = Get_posemap(pose, Human, 64, 1)
			# print(pose)
			# fig = plt.figure()
			# plt.imshow(pose_heat_map, interpolation='none')
			# fig.savefig('temp.png', dpi=fig.dpi)
			# plt.imshow(I)
			# plt.axis('off')
			# ax = plt.gca()
			# coco_kps.showAnns(a)
			# plt.show()
			# fig.savefig('temp2.png', dpi=fig.dpi)
			flag += 1
			# print("-----------------------")
			# GT.append((pose_heat_map, pose_mask))
			GT.append(pose)

	assert(flag==1)	


# for imageid in Trainval_GT:
# 	for GT in Trainval_GT[imageid]:
# 		Image_id = GT[0]
# 		Human = GT[2]
# 		annotations_coco = coco_kps.loadAnns(coco_kps.getAnnIds(imgIds=Image_id))
# 		flag = 0
# 		mykeypoints = None
# 		mindist = None
# 		for a in annotations_coco:
# 			bbox = np.array(a['bbox'])
# 			if mindist == None:
# 				mindist = np.sum((bbox - Human)**2)
# 				mykeypoints = normalize_human_pose(a['keypoints'])
# 			elif mindist > np.sum((bbox - Human)**2):
# 				mindist = np.sum((bbox - Human)**2)
# 				mykeypoints = normalize_human_pose(a['keypoints'])
# 			else:
# 				continue

# 		if np.sum(mykeypoints) == 0:
# 			zero_poses_count += 1

# 		total_poses += 1
# 		GT.append(mykeypoints)

pkl.dump(Trainval_GT, open( data_dir + input_pkl_file + '_Keypoints_just_pose.pkl', "wb"), protocol=pkl.HIGHEST_PROTOCOL)
print("Dumping into ", data_dir + input_pkl_file + '_Keypoints_just_pose.pkl')
# print("Num Zero poses : ", zero_poses_count)
print("Total Poses : ", total_poses)