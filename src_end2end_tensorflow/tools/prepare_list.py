import os
import sys
import json
import scipy.io as sio
import pickle as pkl
import h5py
import numpy as np
import pdb

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
root = '/home/rishabh/scene_graph/hoi_graph/v-coco-master/' # vsrl toolkit dir
add_path(root)
add_path(root+'coco/PythonAPI/')
import vsrl_utils as vu

dataset = 'hico'

data_dir = '/home/hoi_graph/data'
vrd_dir = data_dir + '/VRD/dataset' # VRD dir
hico_dir = data_dir + '/hico_20160224_det' # HICO_DET dir
vcoco_dir = data_dir + '/v-coco' # V-COCO dir

hoi_list = {0: u'surf_instr', 1: u'ski_instr', 2: u'cut_instr', 3: u'walk', 4: u'cut_obj', 5: u'ride_instr', 6: u'talk_on_phone_instr', 7: u'kick_obj', 8: u'work_on_computer_instr', 9: u'eat_obj', 10: u'sit_instr', 11: u'jump_instr', 12: u'lay_instr', 13: u'drink_instr', 14: u'carry_obj', 15: u'throw_obj', 16: u'eat_instr', 17: u'smile', 18: u'look_obj', 19: u'hit_instr', 20: u'hit_obj', 21: u'snowboard_instr', 22: u'run', 23: u'point_instr', 24: u'read_obj', 25: u'hold_obj', 26: u'skateboard_instr', 27: u'stand', 28: u'catch_obj'}

# Save object and predicate in word_embedding_input.json
words = []

# VRD
objectListN_file = os.path.join(vrd_dir, 'objectListN.mat') 
predicate = os.path.join(vrd_dir, 'predicate.mat')
objects = sio.loadmat(objectListN_file, struct_as_record=False, squeeze_me=True)['objectListN'] 
predicates = sio.loadmat(predicate, struct_as_record=False, squeeze_me=True)['predicate'] 
for obj in objects:
    if obj not in words:
        words.append(str(obj))
for pre in predicates:
    if pre not in words:
        words.append(str(pre))
        

if dataset=='hico':
    objectListN_file = os.path.join(hico_dir, 'devkit/objectListN.mat')
    predicates_file = os.path.join(hico_dir, 'devkit/predicate.mat')
    objects_hico = sio.loadmat(objectListN_file, struct_as_record=False, squeeze_me=True)['objectListN'] 
    predicates_hico = sio.loadmat(predicates_file, struct_as_record=False, squeeze_me=True)['predicate'] 
    for obj in objects_hico:
        if obj not in words:
            if str(obj)=='baseball_glove\\':
                words.append('baseball_glove')
            else:
                words.append(str(obj))
    for pre in predicates_hico:
        if pre not in words:
            words.append(str(pre))
    save_file = os.path.join(data_dir, 'list', 'words_hico_vrd.json')
            
if dataset=='vcoco':
    objectListN_file = os.path.join(vcoco_dir, 'devkit/cats.mat')
    predicates_file = os.path.join(vcoco_dir, 'devkit/actions.mat')
    tmp = sio.loadmat(objectListN_file, struct_as_record=False, squeeze_me=True)['cats'] 
    objects_vcoco = [tmp[i].name for i in range(len(tmp))]
    predicates_vcoco = sio.loadmat(predicates_file, struct_as_record=False, squeeze_me=True)['action']
    for obj in objects_vcoco:
        if obj not in words:
            words.append(str(obj))
    for pre in predicates_vcoco:
        if pre not in words:
            words.append(str(pre)) 
    
    save_file = os.path.join(data_dir, 'list', 'words_vcoco_vrd.json')

with open(save_file, 'w') as fp:
    json.dump(words, fp)
    print('Save vertice words in text to %s' % save_file)

# Save internal+vrd_graph.pkl 
# VRD
def extract_vrd(split, triplets, edges):
    anno_file = os.path.join(vrd_dir, 'annotation_%s.mat'%split) 
    anno = sio.loadmat(anno_file, struct_as_record=False, squeeze_me=True)['annotation_%s'%split]   
    for i in range(len(anno)):
        if 'relationship' not in anno[i]._fieldnames:
            continue
        relations = anno[i].relationship
        if not type(relations)==np.ndarray:
            relations = [relations]

        for j in range(len(relations)):
            rel = relations[j].phrase
            trip = list(rel)
            if not trip in triplets:
                triplets.append(trip) 
            edge1 = list([trip[0], trip[1]])
            edge2 = list([trip[1], trip[2]])
            if edge1 not in edges:
                edges.append(edge1)
            if edge2 not in edges:
                edges.append(edge2) 
    return triplets, edges

triplets = []
edges = []

triplets, edges = extract_vrd('train', triplets, edges)
triplets, edges = extract_vrd('test', triplets, edges)


def extract_vcoco(split, edges):
    anno_file = '/media/SeSaMe_NAS/data/v-coco/mydata/annotation/%s.json'%split
    with open(anno_file) as json_data:
        rdata = json.load(json_data)
    meta = h5py.File('/media/SeSaMe_NAS/data/v-coco/devkit/vcoco_meta.h5','r')
    humans = []
    objects = []
    relations = []
    for imid in rdata.keys():
        anno = rdata[imid]
        for r in range(len(anno)):
            human = []
            obj = []
            objnames = []
            relation = np.zeros((26))
            if anno[r]['label'] and not np.isnan(anno[r]['role_bbox'][4:]).all():
                predicate = anno[r]['action_name']
                human.append(anno[r]['bbox'])
                role_names = anno[r]['role_name']
                role_ids = anno[r]['role_object_id']

                if len(anno[r]['role_name'])==2:
                    obj.append(anno[r]['role_bbox'][4:])                
                    objnames.append(vu.coco_obj_id_to_obj_class(int(role_ids[1]), coco))
                elif len(anno[r]['role_name'])==3:
                    if not np.isnan(anno[r]['role_bbox'][4:8]).any():
                        obj.append(anno[r]['role_bbox'][4:8])
                        objnames.append(vu.coco_obj_id_to_obj_class(int(role_ids[1]), coco))
                    if not np.isnan(anno[r]['role_bbox'][8:]).any():    
                        obj.append(anno[r]['role_bbox'][8:])
                        objnames.append(vu.coco_obj_id_to_obj_class(int(role_ids[2]), coco))
                aid = meta['meta/pre/name2idx/' + predicate][...]
                aid = aid.astype(int)
                relation[aid] = 1 

                for objname in objnames:
                    #edge1 = list(['person', predicate])
                    edge2 = list([predicate, objname])
                    #if edge1 not in edges:
                    #    edges.append(edge1)
                    if edge2 not in edges:
                        edges.append(edge2) 
    return edges

if dataset=='vcoco':
    coco = vu.load_coco()    
    edges = extract_vcoco('trainval', edges)
    edges = extract_vcoco('test', edges)
    save_file = os.path.join(data_dir, 'list', 'words_vcoco.json')    
    with open(save_file) as fp:
            words = json.load(fp)

    ver_dict = {}
    graph = {} 
    for i in range(len(words)): 
        ver_dict[words[i]] = i
        graph[i] = []   
    for i in range(len(edges)):
        if not (ver_dict.has_key(edges[i][0]) and ver_dict.has_key(edges[i][1])):
            print('no!!!', i)
        id1 = ver_dict[edges[i][0]]
        id2 = ver_dict[edges[i][1]]
        graph[id1].append(id2) # undirected
        graph[id2].append(id1)

    graph_file = os.path.join(data_dir, 'vcoco_vrd_graph.pkl')
    with open(graph_file, 'wb') as fp:
        pkl.dump(graph, fp)
        print('Save VCOCO + VRD graph structure to: ', graph_file)     
        
def extract_hico(split, edges, anno, list_action, meta):
    humans = []
    objects = []
    relations = []
    onames = []
    bboxes = anno['bbox_%s'%split]
    nimgs = len(bboxes)
    for idx in range(nimgs):
        bbox = bboxes[idx] # 'filename',' hoi', 'size'
        hois = bbox.hoi # 'id', 'bboxhuman', 'bboxobject', 'connection', 'invis'

        if not type(hois)==np.ndarray: 
            hois = [hois]
        for hoi in hois:
            relation = np.zeros((117))
            if hoi.invis: continue      
            hid = hoi.id-1 # 0-599 in python
            predicate = list_action[hid].vname
            oname = list_action[hid].nname

            #edge1 = list(['person', predicate])
            edge2 = list([predicate, oname])
            #if edge1 not in edges:
            #    edges.append(edge1)
            if edge2 not in edges:
                edges.append(edge2)  
     
    return edges

if dataset=='hico':
    anno_file = os.path.join(hico_dir, 'devkit/anno_bbox.mat')
    anno = sio.loadmat(anno_file, struct_as_record=False, squeeze_me=True)
    list_action = anno['list_action'] # len=600
    meta_file = os.path.join(hico_dir, 'devkit/hico_det_meta.h5')
    meta = h5py.File(meta_file, 'r')   
    edges = extract_hico('train', edges, anno, list_action, meta)
    edges = extract_hico('test', edges, anno, list_action, meta)
    
    # matched with relationship annotation text
    save_file = os.path.join(data_dir, 'list', 'words_hico_vrd.json')
    with open(save_file) as fp:
        words = json.load(fp)

    ver_dict = {}
    graph = {} 
    for i in range(len(words)):
        ver_dict[words[i]] = i
        graph[i] = []   
    for i in range(len(edges)):
        if not (ver_dict.has_key(edges[i][0]) and ver_dict.has_key(edges[i][1])):
            print('no!!!', i)
        id1 = ver_dict[edges[i][0]]
        id2 = ver_dict[edges[i][1]]
        graph[id1].append(id2) # undirected
        graph[id2].append(id1)

    graph_file = os.path.join(data_dir, 'hico_vrd_graph.pkl')
    with open(graph_file, 'wb') as fp:
        pkl.dump(graph, fp)
        print('Save HICO-DET + VRD graph structure to: ', graph_file) 
        
