


import os
import random
from math import radians

import numpy as np 
from PIL import Image, ImageOps
import cv2
import torch
import json
import trimesh  
import logging

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

from ..geometry import index, orthogonal

from numpy.linalg import inv
from scipy import sparse  

NUM_OF_FACES = 6
NUM_OF_INITIAL_FEATURES = 7

GT_OUTPUT_RESOLUTION = 512 
GT_NUM_OF_FACES = 6
GT_NUM_OF_FEATURES = 4 

log = logging.getLogger('trimesh')
log.setLevel(40)



def load_trimesh(root_dir, training_subject_list = None):

    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        if f == ".DS_Store" or f == "val.txt":
            continue
        sub_name = f

        if sub_name not in training_subject_list: # only load meshes that are in the training set
            continue

        meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s_clean.obj' % sub_name))
        print(sub_name)

    return meshs



def make_rotate(rx, ry, rz):
    # rx is rotation angle about the x-axis
    # ry is rotation angle about the y-axis
    # rz is rotation angle about the z-axis

    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R


class THumanDataset(Dataset):


    def __init__(self, opt, projection='orthogonal', phase = 'train', evaluation_mode=False, validation_mode=False, frontal_only=False, frontal_only_but_enforce_multi_angles_still=False, use_all_subjects=False, must_generate_sample_pts=False, dynamic_load_mesh=False, return_multiple_sets_of_samples=False):
        self.opt = opt
        self.projection_mode = projection

        use_fake_training_set = True # For debugging only. Set to True to use fake training set

        self.training_subject_list = np.loadtxt("{0}/getTestSet/train_set_list_version_2.txt".format(self.opt.home_dir), dtype=str)

        self.validation_mode = validation_mode
        self.phase = phase
        self.is_train = (self.phase == 'train')

        self.frontal_only = frontal_only
        self.must_generate_sample_pts = must_generate_sample_pts
        self.dynamic_load_mesh = dynamic_load_mesh
        self.return_multiple_sets_of_samples = return_multiple_sets_of_samples

        if self.opt.useValidationSet:

            indices = np.arange( len(self.training_subject_list) )
            np.random.seed(10)
            np.random.shuffle(indices)
            lower_split_index = round( len(self.training_subject_list)* 0.1 )
            val_indices = indices[:lower_split_index]
            train_indices = indices[lower_split_index:]

            if self.validation_mode:
                self.training_subject_list = self.training_subject_list[val_indices]
                self.is_train = False
            else:
                self.training_subject_list = self.training_subject_list[train_indices]

        self.training_subject_list = self.training_subject_list.tolist()


        self.evaluation_mode = evaluation_mode

        if evaluation_mode:
            print("Overwriting self.training_subject_list!")
            self.training_subject_list = np.loadtxt("{0}/getTestSet/test_set_list_version_2.txt".format(self.opt.home_dir), dtype=str).tolist()
            self.is_train = False



        if use_all_subjects:
            print("Overwriting self.training_subject_list!")
            temp_train_and_val_subjects = np.loadtxt("{0}/getTestSet/train_set_list_version_2.txt".format(self.opt.home_dir), dtype=str).tolist()
            temp_test_subjects = np.loadtxt("{0}/getTestSet/test_set_list_version_2.txt".format(self.opt.home_dir), dtype=str).tolist()
            self.training_subject_list = temp_train_and_val_subjects + temp_test_subjects



        if use_fake_training_set:
            self.training_subject_list = np.loadtxt("{0}/getTestSet/newer_fake_train_set_list.txt".format(self.opt.home_dir), dtype=str).tolist()            
            print("using fake training subject list!")



        # use THuman dataset
        self.normal_directory_high_res = "{0}/FSS/trained_normal_maps".format(self.opt.home_dir)
        self.root = "{0}/render_THuman_with_blender/buffer_fixed_full_mesh_sampleData".format(self.opt.home_dir)


        self.mesh_directory = "{0}/split_mesh/results".format(self.opt.home_dir)

        if (  evaluation_mode  or  (self.frontal_only and not self.validation_mode and not self.must_generate_sample_pts)  or self.dynamic_load_mesh or (self.opt.load_sample_pts_from_disk and not self.validation_mode)  ):
            self.mesh_dic = {} 
        else:
            self.mesh_dic = load_trimesh(self.mesh_directory,  training_subject_list = self.training_subject_list)  # a dict containing the meshes of all the CAD models.



        self.frontal_angle_dict = {}



        if self.opt.SDF_Filter:
            self.sdf_plane_directory = "{0}/generate_GT_mesh_SDF/actual_sdf_plane_results".format(self.opt.home_dir)


        if self.opt.useDOS and self.opt.useDOS_ButWithSmplxGuide:
            self.gt_smplx_mesh_directory = "{0}/gt_smplx_in_gt_scale".format(self.opt.home_dir)

        self.subjects = self.training_subject_list 
        #self.subjects = sorted(self.subjects)

        self.num_sample_inout = self.opt.num_sample_inout 


        
        self.img_files = []

        if (not self.frontal_only) or (frontal_only_but_enforce_multi_angles_still) :
            for training_subject in self.subjects:
                subject_render_folder = os.path.join(self.root, training_subject)
                subject_render_paths_list = [  os.path.join(subject_render_folder,f) for f in os.listdir(subject_render_folder) if "image" in f   ]
                self.img_files = self.img_files + subject_render_paths_list
        
        else:
            with open('{0}/split_mesh/all_subjects_with_angles_version_2.txt'.format(self.opt.home_dir), 'r') as f:
                file_list = f.readlines()
                file_list = [ ( line.split()[0], line.split()[1].replace('\n', '') ) for line in file_list ]

            for subject, frontal_angle in file_list:
                if subject in self.subjects:
                    subject_render_filepath = os.path.join(self.root, subject, 'rendered_image_{0:03d}.png'.format( int(frontal_angle) ) )
                    self.img_files.append(subject_render_filepath)




        self.img_files = sorted(self.img_files)


        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(), #  ToTensor converts input to a shape of (C x H x W) in the range [0.0, 1.0] for each dimension
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise with mean of 0.5 and std_dev of 0.5 for each dimension. Finally range will be [-1,1] for each dimension
        ])


        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])


    def __len__(self):
        return len(self.img_files)


    def select_sampling_method_from_disk(self, subject, calib, b_min, b_max, R = None, angle=None):

        samples_low_res_pifu = 0 
        labels_low_res_pifu = 0
        samples_high_res_pifu = 0 
        labels_high_res_pifu = 0


        if self.opt.useDOS and self.opt.useDOS_ButWithSmplxGuide: # SmplxGuide does not use self.opt.num_sample_inout
            set_num_chosen = np.random.randint( low=0 , high=self.opt.num_of_sets_to_sample)
            
            samples_low_res_pifu = np.load('{3}/FSS/apps/results/stored_FSS_sample_pts_forSampleSubjects/subject_{0}_angle_{1:03d}/samples_low_res_pifu_subject_{0}_angle_{1:03d}_set_{2}.npy'.format(subject, angle, set_num_chosen, self.opt.home_dir) ) # (3, num_of_pt)
            labels_low_res_pifu = np.load('{3}/FSS/apps/results/stored_FSS_sample_pts_forSampleSubjects/subject_{0}_angle_{1:03d}/labels_low_res_pifu_subject_{0}_angle_{1:03d}_set_{2}.npy'.format(subject, angle, set_num_chosen, self.opt.home_dir) ) # (1, num_of_pt)

            indices = np.arange( samples_low_res_pifu.shape[1] )
            np.random.shuffle(indices)
            selected_indices = indices[:self.opt.num_sample_inout]
            samples_low_res_pifu = samples_low_res_pifu[:, selected_indices]# [3, self.opt.num_sample_inout]
            labels_low_res_pifu = labels_low_res_pifu[:, selected_indices]# [1, self.opt.num_sample_inout]

        else:

            # spatial sampling
            samples_low_res_pifu = np.load('{2}/FSS/apps/results/stored_SpatialSampling_sample_pts_forSampleSubjects(Only20kPoints)/samples_low_res_pifu_subject_{0}_angle_{1:03d}.npy'.format(subject, angle, self.opt.home_dir) ) # (3, 200000)
            labels_low_res_pifu = np.load('{2}/FSS/apps/results/stored_SpatialSampling_sample_pts_forSampleSubjects(Only20kPoints)/labels_low_res_pifu_subject_{0}_angle_{1:03d}.npy'.format(subject, angle, self.opt.home_dir) ) # (1, 200000)

            indices = np.arange( samples_low_res_pifu.shape[1] )
            np.random.shuffle(indices)
            selected_indices = indices[:self.opt.num_sample_inout]
            samples_low_res_pifu = samples_low_res_pifu[:, selected_indices]# [3, self.opt.num_sample_inout]
            labels_low_res_pifu = labels_low_res_pifu[:, selected_indices]# [1, self.opt.num_sample_inout]


        if self.opt.predict_vertex_normals: 
            vertex_normals_sample_pts = np.load('{2}/FSS/apps/results/stored_PredNormal_sample_pts_forSampleSubjects(Only20kPoints)/vertex_normals_sample_pts_subject_{0}_angle_{1:03d}.npy'.format(subject, angle, self.opt.home_dir) ) # (3, 200000)
            all_vertex_normals_labels = np.load('{2}/FSS/apps/results/stored_PredNormal_sample_pts_forSampleSubjects(Only20kPoints)/all_vertex_normals_labels_subject_{0}_angle_{1:03d}.npy'.format(subject, angle, self.opt.home_dir) ) # (3, 200000)
            
            indices = np.arange( vertex_normals_sample_pts.shape[1] )
            np.random.shuffle(indices)
            selected_indices = indices[:self.opt.num_sample_inout]
            vertex_normals_sample_pts = vertex_normals_sample_pts[:, selected_indices]# [3, self.opt.num_sample_inout]
            all_vertex_normals_labels = all_vertex_normals_labels[:, selected_indices]# [3, self.opt.num_sample_inout]

        else:
            vertex_normals_sample_pts = 0
            all_vertex_normals_labels = 0





        return {
            'samples_low_res_pifu': samples_low_res_pifu,
            'samples_high_res_pifu': samples_high_res_pifu,
            'labels_low_res_pifu': labels_low_res_pifu,
            'labels_high_res_pifu': labels_high_res_pifu,
            'all_vertex_normals_labels': all_vertex_normals_labels,
            'vertex_normals_sample_pts': vertex_normals_sample_pts
            }


    def select_sampling_method(self, subject, calib, b_min, b_max, R = None ):
        samples_high_res_pifu_additional = None
        labels_high_res_pifu_additional = None

        compensation_factor = 4.0

        if self.dynamic_load_mesh and (subject not in self.mesh_dic.keys() ):
            print("loading mesh {0}".format(subject) )
            self.mesh_dic[subject] = trimesh.load(os.path.join(self.mesh_directory, subject, '%s_clean.obj' % subject))
        mesh = self.mesh_dic[subject] # the mesh of 1 subject/CAD

        # note, this is the solution for when dataset is "THuman"
        # adjust sigma according to the mesh's size (measured using the y-coordinates)
        #y_length = np.abs(np.max(mesh.vertices, axis=0)[1])  + np.abs(np.min(mesh.vertices, axis=0)[1] )
        y_length = np.abs(  np.max(mesh.vertices, axis=0)[1]  -  np.min(mesh.vertices, axis=0)[1]  )
        sigma_multiplier = y_length/188

        if self.opt.useDOS and self.opt.useDOS_ButWithSmplxGuide:
            max_num_of_pts_in_cut_off_allowed = 450
            num_of_remaining_sample_pts_allowed = 10000 # opt.useDOS_ButWithSmplxGuide does not use opt.num_sample_inout
            #num_of_counter_pts_allowed = 2600
            initial_num_of_counter_pts_allowed = 3600
            num_of_uniform_pts = 600
            surface_points, face_indices = trimesh.sample.sample_surface(mesh, num_of_remaining_sample_pts_allowed )  
            normal_vectors = mesh.face_normals[face_indices] # [num_of_sample_pts, 3]

        else:
            surface_points, face_indices = trimesh.sample.sample_surface(mesh, int(compensation_factor * 4 * self.num_sample_inout) )  # self.num_sample_inout is no. of sampling points and is default to 8000.

        vertex_normals_sample_pts = 0 # default, will be modified later.

        # add random points within image space
        length = b_max - b_min # has shape of (3,)
        if not self.opt.useDOS:
            random_points = np.random.rand( int(compensation_factor * self.num_sample_inout // 4) , 3) * length + b_min # shape of [compensation_factor*num_sample_inout/4, 3]
            surface_points_shape = list(surface_points.shape)
            random_noise = np.random.normal(scale= self.opt.sigma_low_resolution_pifu * sigma_multiplier, size=surface_points_shape)
            sample_points_low_res_pifu = surface_points + random_noise # sample_points are points very near the surface. The sigma represents the std dev of the normal distribution
            sample_points_low_res_pifu = np.concatenate([sample_points_low_res_pifu, random_points], 0) # shape of [compensation_factor*4.25*num_sample_inout, 3]
            np.random.shuffle(sample_points_low_res_pifu)

            inside_low_res_pifu = mesh.contains(sample_points_low_res_pifu) # return a boolean 1D array of size (num of sample points,)
            inside_points_low_res_pifu = sample_points_low_res_pifu[inside_low_res_pifu]

        if self.opt.predict_vertex_normals and not (self.opt.useDOS and not (self.opt.useDOS_ButWithSmplxGuide) ):

            pq = trimesh.proximity.ProximityQuery(mesh)
            directional_vector = np.array([[0.0,0.0,1.0]]) # 1x3
            
            #directional_vector = directional_vector.T # 3x1
            directional_vector = np.matmul(inv(R), directional_vector.T) # 3x1

            z_displacement = np.repeat(directional_vector, self.num_sample_inout, axis=1) # [3, num_of_sample_pts]

            std_dev = 2.5
            dos_compensation_factor = 4.0  
            normal_sigma = np.random.normal(loc=0.0, scale= std_dev  , size= [ int(dos_compensation_factor * self.num_sample_inout) , 1] ) # shape of [num_of_sample_pts, 1]
            normal_sigma_mask = (normal_sigma[:,0] < 1.0)  &  (normal_sigma[:,0] > -1.0)
            normal_sigma = normal_sigma[normal_sigma_mask,:]
            while(normal_sigma.shape[0] < self.num_sample_inout):
                print("Warning: Not enough normal sigma labels accepted!")
                normal_sigma = np.random.normal(loc=0.0, scale= std_dev  , size= [ int(dos_compensation_factor * self.num_sample_inout) , 1] ) # shape of [num_of_sample_pts, 1]
                normal_sigma_mask = (normal_sigma[:,0] < 1.0)  &  (normal_sigma[:,0] > -1.0)
                normal_sigma = normal_sigma[normal_sigma_mask,:]

            normal_sigma = normal_sigma[0:self.num_sample_inout, :]

            if self.opt.vertex_normals_config in [2,3]:
                vertex_normals_sample_pts = surface_points[0:self.num_sample_inout, :] + z_displacement.T * sigma_multiplier * normal_sigma * 2.0 
            else:
                vertex_normals_sample_pts = surface_points[0:self.num_sample_inout, :] + z_displacement.T * sigma_multiplier * normal_sigma * 3.0 #2.0 

            temp_vertex_normals_sample_pts = surface_points[0:self.num_sample_inout, :]

            positive_proximity = trimesh.proximity.longest_ray(mesh, vertex_normals_sample_pts , z_displacement.T ) # shape of [num_of_pts,]
            negative_proximity = trimesh.proximity.longest_ray(mesh, vertex_normals_sample_pts , -z_displacement.T ) # shape of [num_of_pts,]

            shortest_proximity = np.minimum(positive_proximity, negative_proximity)
            is_inf = np.isinf(shortest_proximity)
            shortest_proximity[is_inf] = 0
            shortest_direction = positive_proximity <= negative_proximity
            shortest_direction = shortest_direction.astype(np.float32)
            shortest_direction[shortest_direction==0] = -1
            shortest_direction = z_displacement * shortest_direction[None, ...]   # [3, num_of_sample_pts]
            shortest_direction = shortest_direction * shortest_proximity[None, ...] # [3, num_of_sample_pts]

            vertex_normals_sample_pts = vertex_normals_sample_pts + shortest_direction.T  # [num_of_pts, 3]
            vertex_normals_sample_pts[is_inf] = temp_vertex_normals_sample_pts[is_inf]

            _, vertex_id = pq.vertex( vertex_normals_sample_pts )
            vertex_normals_labels = mesh.vertex_normals[vertex_id, :] # [num_of_pts, 3]
            vertex_normals_labels = np.matmul(R, vertex_normals_labels.T) # 3 x num_of_pts
            vertex_normals_labels = vertex_normals_labels.T


            vertex_normals_labels[is_inf, :] = 0
            vertex_normals_sample_pts[is_inf, :] = 0  







        if self.opt.useDOS:

   
            if self.opt.useDOS_ButWithSmplxGuide:
                pq = trimesh.proximity.ProximityQuery(mesh)
                std_dev_1 = 1.0
                magnitude_multiplier_1 = 2.0


                gt_smplx_mesh_path = os.path.join(self.gt_smplx_mesh_directory, "gt_smplx_{0}.obj".format(subject) )
                gt_smplx_mesh = trimesh.load(gt_smplx_mesh_path, process=False)
                gt_smplx_vertices = gt_smplx_mesh.vertices

                part_segm = json.load(open( os.path.join(self.gt_smplx_mesh_directory, 'smplx_vert_segmentation.json') ))
                ears_part_segm = json.load(open( os.path.join(self.gt_smplx_mesh_directory, 'smplx_ears_vertex_indices.json') ))
                for (k, v) in part_segm.items():
                    if k == 'rightHand':
                        rightHand_vertices = gt_smplx_vertices[v,:]
                    elif k == 'rightHandIndex1':
                        rightHandIndex1_vertices = gt_smplx_vertices[v,:]
                    elif k == 'leftHand':
                        leftHand_vertices = gt_smplx_vertices[v,:]
                    elif k == 'leftHandIndex1':
                        leftHandIndex1_vertices = gt_smplx_vertices[v,:]

                ears_part_segm = json.load(open( os.path.join( self.gt_smplx_mesh_directory, 'smplx_ears_vertex_indices.json') ))
                for (k, v) in ears_part_segm.items():
                    if k == 'rightEar':
                        rightEar_vertices = gt_smplx_vertices[v,:]
                    elif k == 'leftEar':
                        leftEar_vertices = gt_smplx_vertices[v,:]

                
                desired_vertices = np.concatenate([rightHand_vertices, rightHandIndex1_vertices, leftHand_vertices, leftHandIndex1_vertices, rightEar_vertices, leftEar_vertices ],axis=0)
                
                random_noise_sigma = 2.1 #3.5
                surface_points_shape = list(desired_vertices.shape)
                random_noise = np.random.normal(scale=random_noise_sigma * sigma_multiplier, size=surface_points_shape)

                """
                desired_vertices = desired_vertices + random_noise   # Shape of (2880, 3)
                desired_vertices, _, _ =  pq.on_surface(points=desired_vertices)
                """
                temp_desired_vertices = desired_vertices + random_noise   # Shape of (2880, 3)
                
                random_noise = np.random.normal(scale=random_noise_sigma * sigma_multiplier, size=surface_points_shape) # do it one more time
                desired_vertices = desired_vertices + random_noise   # Shape of (2880, 3)
                desired_vertices = np.concatenate([temp_desired_vertices, desired_vertices], axis=0)
                
                camera_vector = np.array([[0.0,0.0,1.0]]) # 1x3
                camera_vector = np.matmul(inv(R), camera_vector.T) # 3x1
                z_displace = np.repeat(camera_vector, desired_vertices.shape[0], axis=1) # [3, num_of_sample_pts]
                positive_proximity = trimesh.proximity.longest_ray(mesh, desired_vertices , z_displace.T ) # shape of [num_of_pts,]
                negative_proximity = trimesh.proximity.longest_ray(mesh, desired_vertices , -z_displace.T ) # shape of [num_of_pts,]
                shortest_proximity = np.minimum(positive_proximity, negative_proximity)
                is_inf = np.isinf(shortest_proximity)
                shortest_proximity[is_inf] = 0
                shortest_direction = positive_proximity <= negative_proximity
                shortest_direction = shortest_direction.astype(np.float32)
                shortest_direction[shortest_direction==0] = -1
                shortest_direction = z_displace * shortest_direction[None, ...]   # [3, num_of_sample_pts]
                final_dist_to_displace = shortest_direction * shortest_proximity[None, ...] # [3, num_of_sample_pts]

                desired_vertices = desired_vertices + final_dist_to_displace.T  # [num_of_pts, 3]
                desired_vertices = desired_vertices[ np.logical_not(is_inf) ]  # [no_of_pts, 3]

                _, closest_vertex_id = pq.vertex( desired_vertices ) # (num_of_pts,)
                normals_of_desired_vertices = mesh.vertex_normals[closest_vertex_id, :] # (num_of_pts, 3)
                
                thickness_proximity = trimesh.proximity.longest_ray(mesh, desired_vertices , -normals_of_desired_vertices ) # shape of [num_of_sample_pts,]
            
                #relative_cut_off = 3.2  
                relative_cut_off =1.25 
                cut_off = relative_cut_off * sigma_multiplier # is somewhere between the 1 percentile and the 2 percentile but nearer to the 2 percentile.
                
                cut_off_bool = thickness_proximity<cut_off
                #special_cut_off_bool = thickness_proximity< (1.0 * sigma_multiplier) # for setting a minimum width/thickness

                temp_surface_points_extremities_1 = desired_vertices[cut_off_bool, :]
                temp_normals_extremities_1 = normals_of_desired_vertices[cut_off_bool, :] 
                temp_surface_points_extremities_3 = temp_surface_points_extremities_1 - temp_normals_extremities_1 * thickness_proximity[cut_off_bool, None] * 0.5
                error_check = mesh.contains(temp_surface_points_extremities_3) # return a boolean 1D array of size (num of sample points,)
                incorrect_pts_bool_arr = np.logical_not(error_check)
                temp_cut_off_bool = cut_off_bool.astype(int)
                temp_indices = temp_cut_off_bool.nonzero()[0]
                indices_to_rectify = temp_indices[incorrect_pts_bool_arr] 
                cut_off_bool[indices_to_rectify] = False # set the incorrect pts to false

                # control the number of points that are in the cut-off
                curr_num_of_pts_in_cut_off = np.sum( cut_off_bool.astype(int) )
                if curr_num_of_pts_in_cut_off > max_num_of_pts_in_cut_off_allowed :
                    surplus = curr_num_of_pts_in_cut_off - max_num_of_pts_in_cut_off_allowed
                    arr_indices = np.arange( cut_off_bool.shape[0] )
                    in_indices = arr_indices[cut_off_bool]
                    surplus_in_indices = in_indices[0:surplus]
                    cut_off_bool[surplus_in_indices] = False


                #special_cut_off_bool = np.logical_and(special_cut_off_bool, cut_off_bool)
                not_cut_off_bool = np.logical_not(cut_off_bool)
                num_of_cut_off_pts = cut_off_bool.sum()
                num_of_pts_count_increment = num_of_cut_off_pts * 5
                num_of_counter_pts_allowed = initial_num_of_counter_pts_allowed
                if num_of_pts_count_increment < (max_num_of_pts_in_cut_off_allowed*5):
                    deficit_count = (max_num_of_pts_in_cut_off_allowed*5) - num_of_pts_count_increment
                    """
                    num_of_counter_pts_allowed = num_of_counter_pts_allowed + deficit_count
                    if num_of_counter_pts_allowed % 2 == 1:
                        num_of_uniform_pts += 1
                    """
                    num_of_uniform_pts = num_of_uniform_pts + deficit_count 
                else:
                    pass 



                # 1 == on the front surface; 2 == on the rear surface; 3 == the middle point between 1 and 2; 
                surface_points_extremities_1 = desired_vertices[cut_off_bool, :]
                normals_extremities_1 = normals_of_desired_vertices[cut_off_bool, :] 

                surface_points_extremities_2 =  surface_points_extremities_1 - normals_extremities_1 * thickness_proximity[cut_off_bool, None]
                surface_points_extremities_3 = surface_points_extremities_1 - normals_extremities_1 * thickness_proximity[cut_off_bool, None] * 0.5
                surface_points_extremities_3_child1 = surface_points_extremities_1 - normals_extremities_1 * thickness_proximity[cut_off_bool, None] * 0.25
                surface_points_extremities_3_child2 = surface_points_extremities_1 - normals_extremities_1 * thickness_proximity[cut_off_bool, None] * 0.75

                #special_cut_off_bool = thickness_proximity[cut_off_bool] < (1.0 * sigma_multiplier) # for setting a minimum width/thickness


                excluded_desired_vertices = desired_vertices[not_cut_off_bool, :]
                normals_of_excluded_desired_vertices = normals_of_desired_vertices[not_cut_off_bool, :] 

                #_, vertex_id = pq.vertex( surface_points_extremities_2 ) # (num_of_pts,)
                #normals_extremities_2 = mesh.vertex_normals[vertex_id, :] # [num_of_pts, 3]
                normals_extremities_2 = - normals_extremities_1

                normal_sigma_2 = np.random.normal(loc=0.0, scale= std_dev_1 , size= [ surface_points_extremities_1.shape[0] , 1] ) # shape of [surface_points_extremities_1.shape[0], 1]
                normal_sigma_2_mask = (normal_sigma_2[:,0] < 1.0)  &  (normal_sigma_2[:,0] > -1.0)
                normal_sigma_2 = normal_sigma_2[normal_sigma_2_mask,:]
                normal_sigma_2 = np.abs(normal_sigma_2)
                while(normal_sigma_2.shape[0] < surface_points_extremities_1.shape[0]):
                    #print("Warning: Not enough normal sigma labels accepted!")
                    curr_normal_sigma_2 = np.random.normal(loc=0.0, scale= std_dev_1  , size= [ surface_points_extremities_1.shape[0] , 1] ) # shape of [surface_points_extremities_1.shape[0], 1]
                    curr_normal_sigma_2_mask = (curr_normal_sigma_2[:,0] < 1.0)  &  (curr_normal_sigma_2[:,0] > -1.0)
                    curr_normal_sigma_2 = curr_normal_sigma_2[curr_normal_sigma_2_mask,:]
                    curr_normal_sigma_2 = np.abs(curr_normal_sigma_2)
                    normal_sigma_2 = np.concatenate([normal_sigma_2, curr_normal_sigma_2], axis=0)
                normal_sigma_2 = normal_sigma_2[:surface_points_extremities_1.shape[0], :]


                normal_sigma_for_excluded_desired_vertices = np.random.normal(loc=0.0, scale= std_dev_1 , size= [ excluded_desired_vertices.shape[0] , 1] ) # shape of [excluded_desired_vertices.shape[0], 1]
                normal_sigma_for_excluded_desired_vertices_mask = (normal_sigma_for_excluded_desired_vertices[:,0] < 1.0)  &  (normal_sigma_for_excluded_desired_vertices[:,0] > -1.0)
                normal_sigma_for_excluded_desired_vertices = normal_sigma_for_excluded_desired_vertices[normal_sigma_for_excluded_desired_vertices_mask,:]
                normal_sigma_for_excluded_desired_vertices = np.abs(normal_sigma_for_excluded_desired_vertices)
                while(normal_sigma_for_excluded_desired_vertices.shape[0] < excluded_desired_vertices.shape[0]):
                    #print("Warning: Not enough normal sigma labels accepted!")
                    curr_normal_sigma_for_excluded_desired_vertices = np.random.normal(loc=0.0, scale= std_dev_1  , size= [ excluded_desired_vertices.shape[0] , 1] ) # shape of [excluded_desired_vertices.shape[0], 1]
                    curr_normal_sigma_for_excluded_desired_vertices_mask = (curr_normal_sigma_for_excluded_desired_vertices[:,0] < 1.0)  &  (curr_normal_sigma_for_excluded_desired_vertices[:,0] > -1.0)
                    curr_normal_sigma_for_excluded_desired_vertices = curr_normal_sigma_for_excluded_desired_vertices[curr_normal_sigma_for_excluded_desired_vertices_mask,:]
                    curr_normal_sigma_for_excluded_desired_vertices = np.abs(curr_normal_sigma_for_excluded_desired_vertices)
                    normal_sigma_for_excluded_desired_vertices = np.concatenate([normal_sigma_for_excluded_desired_vertices, curr_normal_sigma_for_excluded_desired_vertices], axis=0)
                normal_sigma_for_excluded_desired_vertices = normal_sigma_for_excluded_desired_vertices[:excluded_desired_vertices.shape[0], :]


                surface_points_extremities_1_out = surface_points_extremities_1 + normals_extremities_1 * normal_sigma_2 * thickness_proximity[cut_off_bool, None] * 0.45 # "thickness_proximity[cut_off_bool, None] * 0.45"  is max displacement allowed 
                surface_points_extremities_1_in = surface_points_extremities_1 - normals_extremities_1 * normal_sigma_2 * thickness_proximity[cut_off_bool, None] * 0.45 # "thickness_proximity[cut_off_bool, None] * 0.45"  is max displacement allowed 

                surface_points_extremities_1_out = surface_points_extremities_1_out + normals_extremities_1 * normal_sigma_2 * sigma_multiplier # for min width

                surface_points_extremities_2_out = surface_points_extremities_2 + normals_extremities_2 * normal_sigma_2 * thickness_proximity[cut_off_bool, None] * 0.45 
                surface_points_extremities_2_in = surface_points_extremities_2 - normals_extremities_2 * normal_sigma_2 * thickness_proximity[cut_off_bool, None] * 0.45 

                surface_points_extremities_2_out = surface_points_extremities_2_out + normals_extremities_2  * normal_sigma_2 * sigma_multiplier # for min width

                """
                # experimental only, must comment out!
                surface_points_extremities_1_out = surface_points_extremities_1 + normals_extremities_1 * normal_sigma_2 * sigma_multiplier * magnitude_multiplier_1 # "thickness_proximity[cut_off_bool, None] * 0.45"  is max displacement allowed 
                surface_points_extremities_1_in = surface_points_extremities_1 - normals_extremities_1 * normal_sigma_2 * sigma_multiplier * magnitude_multiplier_1 # "thickness_proximity[cut_off_bool, None] * 0.45"  is max displacement allowed 
                surface_points_extremities_2_out = surface_points_extremities_2 + normals_extremities_2 * normal_sigma_2 * sigma_multiplier * magnitude_multiplier_1  
                surface_points_extremities_2_in = surface_points_extremities_2 - normals_extremities_2 * normal_sigma_2 * sigma_multiplier * magnitude_multiplier_1  
                """



                excluded_desired_vertices_out = excluded_desired_vertices + normals_of_excluded_desired_vertices * (normal_sigma_for_excluded_desired_vertices) * sigma_multiplier * magnitude_multiplier_1
                excluded_desired_vertices_in = excluded_desired_vertices - normals_of_excluded_desired_vertices * normal_sigma_for_excluded_desired_vertices * sigma_multiplier * magnitude_multiplier_1
                #excluded_desired_vertices_in_additional = excluded_desired_vertices - normals_of_excluded_desired_vertices * (normal_sigma_for_excluded_desired_vertices*0.5) * sigma_multiplier


                # get remaining points
                remaining_surface_points = surface_points # [num_of_remaining_sample_pts, 3]
                remaining_surface_normals = normal_vectors 

                normal_sigma_1 = np.random.normal(loc=0.0, scale= std_dev_1 , size= [ remaining_surface_points.shape[0] , 1] ) # shape of [num_of_remaining_sample_pts, 1]
                normal_sigma_1_mask = (normal_sigma_1[:,0] < 1.0)  &  (normal_sigma_1[:,0] > -1.0)
                normal_sigma_1 = normal_sigma_1[normal_sigma_1_mask,:]
                while(normal_sigma_1.shape[0] < remaining_surface_points.shape[0]):
                    #print("Warning: Not enough normal sigma labels accepted!")
                    curr_normal_sigma_1 = np.random.normal(loc=0.0, scale= std_dev_1  , size= [ remaining_surface_points.shape[0] , 1] ) # shape of [num_of_remaining_sample_pts, 1]
                    curr_normal_sigma_1_mask = (curr_normal_sigma_1[:,0] < 1.0)  &  (curr_normal_sigma_1[:,0] > -1.0)
                    curr_normal_sigma_1 = curr_normal_sigma_1[curr_normal_sigma_1_mask,:]
                    normal_sigma_1 = np.concatenate([normal_sigma_1, curr_normal_sigma_1], axis=0)

                normal_sigma_1 = normal_sigma_1[:remaining_surface_points.shape[0], :]
                remaining_surface_points = remaining_surface_points + remaining_surface_normals * sigma_multiplier * normal_sigma_1 * magnitude_multiplier_1  # shape of [num_of_remaining_sample_pts, 3]



                # get counter points
                counter_pt_indices = np.arange( surface_points.shape[0] )
                np.random.shuffle(counter_pt_indices)
                temp_num_of_counter_pts_allowed = num_of_counter_pts_allowed//2
                selected_counter_pt_indices = counter_pt_indices[:temp_num_of_counter_pts_allowed]
                counter_points = surface_points[selected_counter_pt_indices]# [num_of_counter_pts_allowed//2, 3]
                counter_points_normals = normal_vectors[ selected_counter_pt_indices ,:]

                camera_vector = np.array([[0.0,0.0,1.0]]) # 1x3
                camera_vector = np.matmul(inv(R), camera_vector.T) # 3x1

                dot_product = np.matmul(camera_vector.T, counter_points_normals.T ) # [1 x num_of_sample_pts]
                dot_product[dot_product<0] = -1.0 # points generated from faces that are facing backwards
                dot_product[dot_product>=0] = 1.0 # points generated from faces that are facing camera

                z_displacement = np.matmul(dot_product.T, camera_vector.T) # [num_of_sample_pts, 3]. Will displace points facing backwards to go backwards, but points facing forward to go forward

                displacement_for_counter_points = sigma_multiplier * magnitude_multiplier_1 
                uniform_noise = np.random.uniform(low=3.5, high=4.0, size=(counter_points.shape[0], 1) )
                counter_points = counter_points + z_displacement * displacement_for_counter_points * uniform_noise  # shape of [num_of_remaining_sample_pts, 3]

                secondary_uniform_noise = np.random.uniform(low=1.0, high=2.0, size=(counter_points.shape[0], 1) )
                secondary_counter_points = counter_points + z_displacement * displacement_for_counter_points * secondary_uniform_noise
                counter_points = np.concatenate( [counter_points, secondary_counter_points], axis=0 ) # shape of [num_of_remaining_sample_pts, 3]
                z_displacement = np.concatenate( [z_displacement, z_displacement], axis=0 )


                # get uniform points
                #print("num_of_cut_off_pts * 5:", num_of_cut_off_pts * 5)
                uniform_points = np.random.rand( num_of_uniform_pts , 3) * length + b_min # shape of [ num_of_pts, 3]
                dummy_uniform_points_normals = np.repeat(camera_vector, uniform_points.shape[0] , axis=1) # (3, num_of_pts)
                dummy_uniform_points_normals = dummy_uniform_points_normals.T # shape of [ num_of_pts, 3]



                # now compute their labels
                
                # What we used to do:
                #all_points_to_use = np.concatenate( [surface_points_extremities_1_out, surface_points_extremities_1_in, surface_points_extremities_2_out, surface_points_extremities_2_in, surface_points_extremities_3, surface_points_extremities_3_child1, surface_points_extremities_3_child2, excluded_desired_vertices_out, excluded_desired_vertices_in, remaining_surface_points, counter_points, uniform_points],  axis=0 ) 
                #normal_direction = np.concatenate([-normals_extremities_1, normals_extremities_1, -normals_extremities_2, normals_extremities_2, normals_extremities_1, normals_extremities_1, -normals_extremities_1, -normals_of_excluded_desired_vertices, normals_of_excluded_desired_vertices, -remaining_surface_normals * normal_sigma_1, -z_displacement, -dummy_uniform_points_normals ], axis=0)
                
                # The best way to do it:
                all_points_to_use = np.concatenate( [surface_points_extremities_1_out, surface_points_extremities_1_in, surface_points_extremities_2_out, surface_points_extremities_2_in, surface_points_extremities_3, surface_points_extremities_3_child1, surface_points_extremities_3_child2, remaining_surface_points, counter_points, uniform_points],  axis=0 ) 
                normal_direction = np.concatenate([-normals_extremities_1, normals_extremities_1, -normals_extremities_2, normals_extremities_2, normals_extremities_1, normals_extremities_1, -normals_extremities_1, -remaining_surface_normals * normal_sigma_1, -z_displacement, -dummy_uniform_points_normals ], axis=0)
                
                # For experimental purpose only:
                #all_points_to_use = np.concatenate( [surface_points_extremities_1_out, surface_points_extremities_1_in, surface_points_extremities_2_out, surface_points_extremities_2_in, surface_points_extremities_3, surface_points_extremities_3_child1, surface_points_extremities_3_child2, remaining_surface_points, uniform_points],  axis=0 ) 
                #normal_direction = np.concatenate([-normals_extremities_1, normals_extremities_1, -normals_extremities_2, normals_extremities_2, normals_extremities_1, normals_extremities_1, -normals_extremities_1, -remaining_surface_normals * normal_sigma_1, -dummy_uniform_points_normals ], axis=0)
                
                #all_points_to_use = np.concatenate( [remaining_surface_points, counter_points, uniform_points],  axis=0 ) 
                #normal_direction = np.concatenate([-remaining_surface_normals * normal_sigma_1, -z_displacement, -dummy_uniform_points_normals ], axis=0)

                #all_points_to_use = np.concatenate( [surface_points_extremities_1_out, surface_points_extremities_1_in, surface_points_extremities_2_out, surface_points_extremities_2_in, remaining_surface_points, counter_points, uniform_points],  axis=0 ) 
                #normal_direction = np.concatenate([-normals_extremities_1, normals_extremities_1, -normals_extremities_2, normals_extremities_2, -remaining_surface_normals * normal_sigma_1, -z_displacement, -dummy_uniform_points_normals ], axis=0)
                



                inside_mesh_bool = None
                proximity_final = None
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        for k in [-1,0,1]:
                            if (i == 0 and j == 0 and k == 0) :
                                continue
                            temp_direction = np.array([[i,j,k]]) # [1,3]
                            temp_direction =  temp_direction / np.linalg.norm(temp_direction)
                            temp_direction = temp_direction.T # [3,1]
                            temp_direction = np.repeat(temp_direction, all_points_to_use.shape[0] , axis=1) # (3, num_of_pts)
                            temp_proximity = trimesh.proximity.longest_ray(mesh, all_points_to_use , temp_direction.T ) # shape of [num_of_pts,]
                            if inside_mesh_bool is None:
                                inside_mesh_bool = temp_proximity
                            else:
                                inside_mesh_bool = np.maximum(inside_mesh_bool, temp_proximity)

                            if proximity_final is None:
                                proximity_final = temp_proximity
                            else:
                                proximity_final = np.minimum(proximity_final, temp_proximity)

                inside_mesh_bool = np.isinf(inside_mesh_bool)
                inside_mesh_bool = np.logical_not(inside_mesh_bool)
                inside_mesh_float = inside_mesh_bool.astype(np.float32)
                inside_mesh_float[inside_mesh_float==0] = -1.0


                normal_proximity = trimesh.proximity.longest_ray(mesh, all_points_to_use , normal_direction ) # shape of [num_of_pts,]
                proximity_final = np.minimum(proximity_final, normal_proximity)


                is_inf_bool_final = np.isinf(proximity_final)
                proximity_final[is_inf_bool_final] = 0.0  

                proximity_final = proximity_final * inside_mesh_float

                upper_limit = 2.0   
                lower_limit = -1.0  

                # adjust for the min width points
                num_of_extremities_member = surface_points_extremities_1_out.shape[0]
                """
                expanded_special_cut_off_bool = np.tile(special_cut_off_bool, 4)
                temp_arr = np.copy( proximity_final[0: (4*num_of_extremities_member) ] )
                temp_arr[expanded_special_cut_off_bool] =  temp_arr[expanded_special_cut_off_bool] + 0.6 * sigma_multiplier
                proximity_final[0: (4*num_of_extremities_member) ] = temp_arr
                """
                proximity_final[0: (7*num_of_extremities_member) ] = proximity_final[0: (7*num_of_extremities_member) ] + 0.10 * sigma_multiplier  # + 0.40 * sigma_multiplier

                all_samplePts_labels = proximity_final / ( sigma_multiplier * magnitude_multiplier_1 ) 
                all_samplePts_labels = all_samplePts_labels / 2.0 * 0.8  #[-0.4,0.4]
                all_samplePts_labels = all_samplePts_labels + 0.5 # [0.1,0.9]

                all_samplePts_labels[is_inf_bool_final] = lower_limit
                all_samplePts_labels[all_samplePts_labels>upper_limit] = upper_limit
                all_samplePts_labels[all_samplePts_labels<lower_limit] = lower_limit
                all_samplePts_labels = all_samplePts_labels[None, ...]  # shape of [1, total_num_of_pts]


                total_points_TwinSDF = all_points_to_use.T
                labels_samplePts_final = all_samplePts_labels



            if self.opt.predict_vertex_normals and (not (self.opt.useDOS_ButWithSmplxGuide) ):

                # compute the barycentric coordinates of each sample
                bary = trimesh.triangles.points_to_barycentric(
                    triangles=mesh.triangles[ face_indices[0:self.num_sample_inout] ], points=surface_points[ 0:self.num_sample_inout, : ] )
                # interpolate vertex normals from barycentric coordinates
                vertex_normals_labels = trimesh.unitize((mesh.vertex_normals[mesh.faces[face_indices[0:self.num_sample_inout]]] *
                                          trimesh.unitize(bary).reshape(
                                              (-1, 3, 1))).sum(axis=1))   # Shape of [self.num_sample_inout, 3]

                vertex_normals_labels = np.matmul(R, vertex_normals_labels.T) # 3 x num_of_pts
                vertex_normals_labels = vertex_normals_labels.T





        else:
            outside_points_low_res_pifu = sample_points_low_res_pifu[np.logical_not(inside_low_res_pifu)]



        # reduce the number of inside and outside points if there are too many inside points. (it is very likely that "nin > self.num_sample_inout // 2" is true)
        if not ( self.opt.useDOS):
            nin = inside_points_low_res_pifu.shape[0]
            inside_points_low_res_pifu = inside_points_low_res_pifu[
                            :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points_low_res_pifu  # should have shape of [2500, 3]
            outside_points_low_res_pifu = outside_points_low_res_pifu[
                             :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points_low_res_pifu[
                                                                                                   :(self.num_sample_inout - nin)]     # should have shape of [2500, 3]

            samples_low_res_pifu = np.concatenate([inside_points_low_res_pifu, outside_points_low_res_pifu], 0).T   # should have shape of [3, 5000]




        
        all_vertex_normals_labels = 0 # default, will be modified later.

        if self.opt.useDOS:


            if self.opt.useDOS_ButWithSmplxGuide:
                samples_low_res_pifu = total_points_TwinSDF # Shape of [3, num_of_pts]
                labels_low_res_pifu = labels_samplePts_final # shape of [1, num_of_pts]

        else:
            labels_low_res_pifu = np.concatenate([np.ones((1, inside_points_low_res_pifu.shape[0])), np.zeros((1, outside_points_low_res_pifu.shape[0]))], 1) # should have shape of [1, 5000]. If element is 1, it means the point is inside. If element is 0, it means the point is outside.

        if self.opt.predict_vertex_normals and not (self.opt.useDOS and not (self.opt.useDOS_ButWithSmplxGuide) ):
            all_vertex_normals_labels = vertex_normals_labels.T # Shape of [3, num_of_pts]
            all_vertex_normals_labels = torch.Tensor(all_vertex_normals_labels).float()
            vertex_normals_sample_pts = vertex_normals_sample_pts.T  # Shape of [3, num_of_pts]
            vertex_normals_sample_pts = torch.Tensor(vertex_normals_sample_pts).float()



        samples_low_res_pifu = torch.Tensor(samples_low_res_pifu).float()
        labels_low_res_pifu = torch.Tensor(labels_low_res_pifu).float()





        samplePts_dict = {
            'samples_low_res_pifu': samples_low_res_pifu,
            'labels_low_res_pifu': labels_low_res_pifu,
            'all_vertex_normals_labels': all_vertex_normals_labels,
            'vertex_normals_sample_pts': vertex_normals_sample_pts
            }


        del mesh

        return samplePts_dict





    def get_item(self, index):

        img_path = self.img_files[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # get yaw
        yaw = img_name.split("_")[-1]
        yaw = int(yaw)


        # get subject
        subject = img_path.split('/')[-2] # e.g. "0507"

        param_path = os.path.join(self.root, subject , "rendered_params_" + "{0:03d}".format(yaw) + ".npy"  )
        render_path = os.path.join(self.root, subject, "rendered_image_" + "{0:03d}".format(yaw) + ".png"  )
        mask_path = os.path.join(self.root, subject, "rendered_mask_" + "{0:03d}".format(yaw) + ".png"  )
        
        nmlF_high_res_path =  os.path.join(self.normal_directory_high_res, subject, "rendered_nmlF_" + "{0:03d}".format(yaw) + ".npz"  )
        nmlB_high_res_path =  os.path.join(self.normal_directory_high_res, subject, "rendered_nmlB_" + "{0:03d}".format(yaw) + ".npz"  )



        sdf_plane = 0
        if self.opt.SDF_Filter and (self.opt.SDF_Filter_config == 1):
            sdf_plane_path = os.path.join( self.sdf_plane_directory,  'sdf_plane_subject_{0}_angle_{1:03d}.npy'.format(subject, yaw)  )
            sdf_plane = np.load(sdf_plane_path) # (128, 128)
            sdf_plane = torch.tensor(sdf_plane).float()

             

        load_size_associated_with_scale_factor = 1024


        # get params
        param = np.load(param_path, allow_pickle=True)  # param is a np.array that looks similar to a dict.  # ortho_ratio = 0.4 , e.g. scale or y_scale = 0.961994278, e.g. center or vmed = [-1.0486  92.56105  1.0101 ]
        center = param.item().get('center') # is camera 3D center position in the 3D World point space (without any rotation being applied).
        R = param.item().get('R')   # R is used to rotate the CAD model according to a given pitch and yaw.
        scale_factor = param.item().get('scale_factor') # is camera 3D center position in the 3D World point space (without any rotation being applied).


        b_range = load_size_associated_with_scale_factor / scale_factor # e.g. 512/scale_factor
        b_center = center
        b_min = b_center - b_range/2
        b_max = b_center + b_range/2

        translate = -center.reshape(3, 1)
        extrinsic = np.concatenate([R, translate], axis=1)  # when applied on the 3D pts, the rotation is done first, then the translation
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

        temp_extrinsic = np.copy(extrinsic)
        
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = 1.0 * scale_factor #2.4851518#1.0   
        scale_intrinsic[1, 1] = -1.0 * scale_factor #-2.4851518#-1.0
        scale_intrinsic[2, 2] = 1.0 * scale_factor  #2.4851518#1.0

        uv_intrinsic = np.identity(4)
        uv_intrinsic[0, 0] = 1.0 / float(load_size_associated_with_scale_factor // 2) # self.opt.loadSizeGlobal == 512 by default. This value must be 512 unless you change the "scale_factor"
        uv_intrinsic[1, 1] = 1.0 / float(load_size_associated_with_scale_factor // 2) # uv_intrinsic[1, 1] is equal to 1/256
        uv_intrinsic[2, 2] = 1.0 / float(load_size_associated_with_scale_factor // 2) 

        intrinsic = np.matmul(uv_intrinsic, scale_intrinsic)
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float() # calib should still work to transform pts into the [-1,1] range even if the input image size actually changes from 512 to 1024.
        extrinsic = torch.Tensor(extrinsic).float()



        

        render = Image.open(render_path).convert('RGB')

        mask = Image.open(mask_path).convert('L') # convert to grayscale (it shd already be grayscale)

        # to reload back from sparse matrix to numpy array:
        sparse_matrix = sparse.load_npz(nmlF_high_res_path).todense()
        nmlF_high_res = np.asarray(sparse_matrix) 
        nmlF_high_res = np.reshape(nmlF_high_res, [3, 1024, 1024] )

        sparse_matrix = sparse.load_npz(nmlB_high_res_path).todense()
        nmlB_high_res = np.asarray(sparse_matrix) 
        nmlB_high_res = np.reshape(nmlB_high_res, [3, 1024, 1024] )



        mask = transforms.ToTensor()(mask).float() # *Should* have a shape of (C,H,W)

        if self.opt.use_augmentation and self.is_train:
            render = self.aug_trans(render)




        render = self.to_tensor(render)  # normalize render from [0,255] to [-1,1]
        render = mask.expand_as(render) * render


        
        # resize the 1024 x 1024 image to 512 x 512 for the low-resolution pifu
        render_low_pifu = F.interpolate(torch.unsqueeze(render,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        render_low_pifu = render_low_pifu[0]
        mask_low_pifu = mask_low_pifu[0]
        
        

        nmlF_high_res = torch.Tensor(nmlF_high_res)
        nmlB_high_res = torch.Tensor(nmlB_high_res)
        nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res
        nmlB_high_res = mask.expand_as(nmlB_high_res) * nmlB_high_res

        nmlF  = F.interpolate(torch.unsqueeze(nmlF_high_res,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        nmlF = nmlF[0]
        nmlB  = F.interpolate(torch.unsqueeze(nmlB_high_res,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        nmlB = nmlB[0]



        if self.evaluation_mode or (self.frontal_only and not self.must_generate_sample_pts):
            sample_data = {'samples_low_res_pifu':0, 'samples_high_res_pifu':0, 'labels_low_res_pifu':0, 'labels_high_res_pifu':0, 'samples_high_res_pifu_additional':0,'labels_high_res_pifu_additional':0, 'all_vertex_normals_labels':0, 'vertex_normals_sample_pts':0, 'samplePts_multi_angles':0, 'labels_samplePts_multi_angles':0, 'color_labels':0, 'color_surface_points':0, 'simplified_positional_encoding_points':0, 'simplified_positional_encoding_labels':0}

        else:

            if self.opt.load_sample_pts_from_disk:
                if self.opt.num_sample_inout: 
                    sample_data = self.select_sampling_method_from_disk(subject, calib, b_min = b_min, b_max = b_max, R = R, angle=yaw)

            else:
                if not self.return_multiple_sets_of_samples:

                    if self.opt.num_sample_inout:  # opt.num_sample_inout has default of 8000
                        sample_data = self.select_sampling_method(subject, calib, b_min = b_min, b_max = b_max, R = R)

                else:
                    sample_data_list = []
                    for i in range(self.opt.num_of_sets_to_sample):
                        sample_data = self.select_sampling_method(subject, calib, b_min = b_min, b_max = b_max, R = R)

                        sample_data_list.append(sample_data)
                        sample_data = {'samples_low_res_pifu':0, 'samples_high_res_pifu':0, 'labels_low_res_pifu':0, 'labels_high_res_pifu':0, 'samples_high_res_pifu_additional':0,'labels_high_res_pifu_additional':0, 'all_vertex_normals_labels':0, 'vertex_normals_sample_pts':0, 'samplePts_multi_angles':0, 'labels_samplePts_multi_angles':0, 'color_labels':0, 'color_surface_points':0, 'simplified_positional_encoding_points':0, 'simplified_positional_encoding_labels':0}






        final_dict = {
                'name': subject,
                'render_path':render_path,
                'render_low_pifu': render_low_pifu,
                'mask_low_pifu': mask_low_pifu,
                'original_high_res_render':render,
                'mask':mask,
                'calib': calib,
                'extrinsic': extrinsic,    
                'samples_low_res_pifu': sample_data['samples_low_res_pifu'],
                'labels_low_res_pifu': sample_data['labels_low_res_pifu'],
                'all_vertex_normals_labels': sample_data['all_vertex_normals_labels'],
                'vertex_normals_sample_pts': sample_data['vertex_normals_sample_pts'],
                'b_min': b_min,
                'b_max': b_max,
                'nmlF': nmlF,
                'nmlB': nmlB,
                'nmlF_high_res':nmlF_high_res,
                'nmlB_high_res':nmlB_high_res,
                'sdf_plane':sdf_plane
                    }



        if self.return_multiple_sets_of_samples:
            final_dict.update( {"sample_data_list":sample_data_list} )



        return final_dict
                



    def __getitem__(self, index):
        return self.get_item(index)













    












