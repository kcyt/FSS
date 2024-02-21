
import sys
import os
import json
import time 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
#os.environ["LRU_CACHE_CAPACITY"] = "1"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

from lib.options import BaseOptions
from lib.data.THuman_dataset import THumanDataset
from lib.mesh_util import save_obj_mesh_with_color, reconstruction
from lib.geometry import index

# Start of option check

gen_for_normals = False # Whether to generate normal-pred sample pts and their labels
dynamic_load_mesh=True
do_not_concat_samples_together = False # Must be True for SmplxGuided and False otherwise (If generating for PredNormal, set 'False' as well.). The num of sets of samples to return is set to 25.

split_into_chunks = False # whether to use chunks or not
num_of_chunks_to_use = 13 # only relevant if we are using 'split_into_chunks'. Is also equal to the number of new processes to initialize

# End of option check

if split_into_chunks:
    print("split_into_chunks:", split_into_chunks)
    print("num_of_chunks_to_use:", num_of_chunks_to_use)

seed = 10 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


parser = BaseOptions()
opt = parser.parse()



opt.useValidationSet = False 
print('opt.useValidationSet is automatically set to False')

opt.num_sample_inout = 200000 #20000 for sample data to be released
print('Modifying opt.num_sample_inout!')

opt.load_sample_pts_from_disk = False
print('opt.load_sample_pts_from_disk is automatically set to False')




results_folder = '%s/%s' % (opt.results_path, opt.name) 
os.makedirs(results_folder , exist_ok=True)

opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
with open(opt_log, 'w') as outfile:
    outfile.write(json.dumps(vars(opt), indent=2))






def generate_sample_pts(_iterable, opt, split_index=-1):

    if split_index != -1: # i.e. we are using chunks
        import logging 
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("{0}/log_split_index_{1}.txt".format( results_folder , str(split_index)) ),
                logging.StreamHandler()
                ]
            )


    train_dataset= THumanDataset(opt, projection='orthogonal', phase = 'train', must_generate_sample_pts=True, use_all_subjects=True, dynamic_load_mesh=dynamic_load_mesh, return_multiple_sets_of_samples=do_not_concat_samples_together)

    previous_subject = None

    for index_to_use in _iterable:


        train_data = train_dataset.get_item(index=index_to_use) 
        # train_data["img"].shape  has shape of [1, 3, 512, 512]

        subject = train_data['name'] # in string
        if split_index != -1: # i.e. we are using chunks
            logging.info("Processing Subject {0}".format(subject) )
        else:
            print("Processing Subject {0}".format(subject) )

        if dynamic_load_mesh:
            if previous_subject is None:
                previous_subject = subject
            elif previous_subject != subject:
                del train_dataset.mesh_dic[previous_subject]
                if split_index != -1: # i.e. we are using chunks
                    logging.info("Previous subject {0} deleted".format(previous_subject) )
                else:
                    print("Previous subject {0} deleted".format(previous_subject) )
                previous_subject = subject

        path_samples_low_res_pifu = '{0}/{1}/samples_low_res_pifu_subject_{2}.npy'.format(opt.results_path, opt.name, subject)
        path_labels_low_res_pifu= '{0}/{1}/labels_low_res_pifu_subject_{2}.npy'.format(opt.results_path, opt.name, subject)
        path_vertex_normals_sample_pts = '{0}/{1}/vertex_normals_sample_pts_subject_{2}.npy'.format( opt.results_path, opt.name, subject)
        path_all_vertex_normals_labels = '{0}/{1}/all_vertex_normals_labels_subject_{2}.npy'.format(opt.results_path, opt.name, subject)

        render_path = train_data['render_path']
        render_path = render_path.split('rendered_image_')[-1]
        render_angle = render_path.replace('.png','')
        path_samples_low_res_pifu = path_samples_low_res_pifu.replace('.npy' , '_angle_{0}.npy'.format(render_angle) )
        path_labels_low_res_pifu = path_labels_low_res_pifu.replace('.npy' , '_angle_{0}.npy'.format(render_angle) )
        path_vertex_normals_sample_pts = path_vertex_normals_sample_pts.replace('.npy' , '_angle_{0}.npy'.format(render_angle) )
        path_all_vertex_normals_labels = path_all_vertex_normals_labels.replace('.npy' , '_angle_{0}.npy'.format(render_angle) )
    
        if not do_not_concat_samples_together:
            samples_low_res_pifu = train_data['samples_low_res_pifu'].cpu().detach().numpy()
            labels_low_res_pifu = train_data['labels_low_res_pifu'].cpu().detach().numpy()
            np.save(path_samples_low_res_pifu,  samples_low_res_pifu) 
            np.save(path_labels_low_res_pifu, labels_low_res_pifu) 

            if gen_for_normals:
                vertex_normals_sample_pts = train_data['vertex_normals_sample_pts'].cpu().detach().numpy()
                all_vertex_normals_labels = train_data['all_vertex_normals_labels'].cpu().detach().numpy()
                np.save(path_vertex_normals_sample_pts,  vertex_normals_sample_pts) 
                np.save(path_all_vertex_normals_labels, all_vertex_normals_labels) 


        else:

            sample_list = train_data['sample_data_list'] # a list
            subject_angle_specific_folder = '{0}/{1}/subject_{2}_angle_{3}'.format(opt.results_path, opt.name, subject, render_angle)
            os.makedirs(subject_angle_specific_folder, exist_ok=True)
            for set_index in range(opt.num_of_sets_to_sample):
                set_path_samples_low_res_pifu = path_samples_low_res_pifu.replace('.npy' , '_set_{0}.npy'.format(set_index) )
                set_path_labels_low_res_pifu = path_labels_low_res_pifu.replace('.npy' , '_set_{0}.npy'.format(set_index) )
                set_path_vertex_normals_sample_pts = path_vertex_normals_sample_pts.replace('.npy' , '_set_{0}.npy'.format(set_index) )
                set_path_all_vertex_normals_labels = path_all_vertex_normals_labels.replace('.npy' , '_set_{0}.npy'.format(set_index) )
                
                set_path_samples_low_res_pifu = set_path_samples_low_res_pifu.replace( '{0}/{1}'.format(opt.results_path, opt.name) ,  subject_angle_specific_folder )
                set_path_labels_low_res_pifu = set_path_labels_low_res_pifu.replace( '{0}/{1}'.format(opt.results_path, opt.name) ,  subject_angle_specific_folder )
                set_path_vertex_normals_sample_pts = set_path_vertex_normals_sample_pts.replace( '{0}/{1}'.format(opt.results_path, opt.name) ,  subject_angle_specific_folder )
                set_path_all_vertex_normals_labels = set_path_all_vertex_normals_labels.replace( '{0}/{1}'.format(opt.results_path, opt.name) ,  subject_angle_specific_folder )

                sample_dict = sample_list[set_index]

                samples_low_res_pifu = sample_dict['samples_low_res_pifu'].cpu().detach().numpy()
                labels_low_res_pifu = sample_dict['labels_low_res_pifu'].cpu().detach().numpy()
                np.save(set_path_samples_low_res_pifu,  samples_low_res_pifu) 
                np.save(set_path_labels_low_res_pifu, labels_low_res_pifu) 

                if gen_for_normals:
                    vertex_normals_sample_pts = sample_dict['vertex_normals_sample_pts'].cpu().detach().numpy()
                    all_vertex_normals_labels = sample_dict['all_vertex_normals_labels'].cpu().detach().numpy()
                    np.save(set_path_vertex_normals_sample_pts,  vertex_normals_sample_pts) 
                    np.save(set_path_all_vertex_normals_labels, all_vertex_normals_labels)             



def process_chunk(split_index, num_of_chunks_to_use, chunk_size, len_to_iterate, opt):

    if split_index < num_of_chunks_to_use:
        _iterable = tqdm(range(chunk_size*(split_index-1), chunk_size*split_index ))
    else: # only for the last split_index
        _iterable = tqdm(range(chunk_size*(split_index-1), len_to_iterate ))
    generate_sample_pts(_iterable, opt, split_index)




# This train_dataset will not be used for the actual sampling of points
# setting 'dynamic_load_mesh' to True here as a hack to prevent loading of meshes in this train_dataset.
train_dataset= THumanDataset(opt, projection='orthogonal', phase = 'train', must_generate_sample_pts=False, use_all_subjects=True, dynamic_load_mesh=True)
len_to_iterate = len(train_dataset)  

if not split_into_chunks:
    _iterable = tqdm(range(0, len_to_iterate ))
    generate_sample_pts(_iterable, opt)
else:
    from multiprocessing import Process
    processes = []

    chunk_size = len_to_iterate//num_of_chunks_to_use

    for split_index in range(1, num_of_chunks_to_use+1):
        #process_chunk(split_index, num_of_chunks_to_use, chunk_size, len_to_iterate, opt) 

        proc = Process(target=process_chunk, args=[split_index, num_of_chunks_to_use, chunk_size, len_to_iterate, opt] )
        proc.start()
        processes.append(proc)


    # if there is outstanding processes, wait for it first.
    for p in processes:
        print("Checking for outstanding processes before exiting..")
        p.join()
    print("No outstanding processes detected... Exiting!")












