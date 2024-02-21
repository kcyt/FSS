

import sys
import os
import json
import time 


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from evaluation_helper_functions import quick_get_chamfer_and_surface_dist
from numpy.linalg import inv
from scipy import sparse  
import torch 

def validate_meshes(opt, val_mesh_paths, validation_dataset_frontal_only, num_samples_to_use, validation_epoch_cd_dist_list, validation_epoch_p2s_dist_list, epoch, validation_graph_path, start_epoch, epoch_to_start_validation, validation_results_path):

    root = "{0}/render_THuman_with_blender/buffer_fixed_full_mesh_version_2".format(opt.home_dir)

    total_chamfer_distance = []
    total_point_to_surface_distance = []
    for val_path in val_mesh_paths:
        subject = val_path.split('_')[-1]
        subject = subject.replace('.obj','')

        if len(subject) == 3: # is the angle, not the subject
            subject = val_path.split('_')[-2]

        temp_GT_mesh = validation_dataset_frontal_only.mesh_dic[subject]
        GT_mesh = temp_GT_mesh.copy() # to avoid contaminating the loaded GT mesh




        try:  
            print('Computing CD and P2S for {0}'.format( os.path.basename(val_path) ) )
            source_mesh = trimesh.load(val_path)


            # Will scale the meshes without using calibs

            # start of scaling the meshes
            x_median_gt = ( np.median(GT_mesh.vertices[:,0]) ) / 2
            x_median_test = ( np.median(source_mesh.vertices[:,0]) ) / 2
            y_median_gt = ( np.median(GT_mesh.vertices[:,1]) ) / 2
            y_median_test = ( np.median(source_mesh.vertices[:,1]) ) / 2
            z_median_gt = ( np.median(GT_mesh.vertices[:,2]) ) / 2
            z_median_test = ( np.median(source_mesh.vertices[:,2]) ) / 2

            translation_matrix_gt = np.identity(4)
            translation_matrix_gt[0,3] = -x_median_gt
            translation_matrix_gt[1,3] = -y_median_gt
            translation_matrix_gt[2,3] = -z_median_gt
            translation_matrix_test = np.identity(4)
            translation_matrix_test[0,3] = -x_median_test
            translation_matrix_test[1,3] = -y_median_test
            translation_matrix_test[2,3] = -z_median_test

            source_mesh.apply_transform(translation_matrix_test)
            GT_mesh.apply_transform(translation_matrix_gt)

            y_range_gt =  np.max(GT_mesh.vertices[:,1])  - np.min(GT_mesh.vertices[:,1]) 
            y_range_test = np.max(source_mesh.vertices[:,1])  - np.min(source_mesh.vertices[:,1])  
            scale_matrix_gt = np.identity(4)
            scale_matrix_test = np.identity(4)
            scale_matrix_gt[0,0] = scale_matrix_gt[1,1] = scale_matrix_gt[2,2] = 1/y_range_gt
            scale_matrix_test[0,0] = scale_matrix_test[1,1] = scale_matrix_test[2,2] = 1/y_range_test

            GT_mesh.apply_transform(scale_matrix_gt)
            source_mesh.apply_transform(scale_matrix_test)
            GT_mesh.apply_transform(inv(translation_matrix_gt))
            source_mesh.apply_transform(inv(translation_matrix_test))
            # end of scaling the meshes


            chamfer_distance, point_to_surface_distance = quick_get_chamfer_and_surface_dist(src_mesh=source_mesh, tgt_mesh=GT_mesh, num_samples=num_samples_to_use, sideview_confidence_box=sideview_confidence_box )
            total_chamfer_distance.append(chamfer_distance)
            total_point_to_surface_distance.append(point_to_surface_distance)

            del GT_mesh

        except Exception as e:
            print(e)
            print('Unable to compute chamfer_distance and/or point_to_surface_distance!')
    
    if len(total_chamfer_distance) == 0:
        average_chamfer_distance = 0
    else:
        average_chamfer_distance = np.mean(total_chamfer_distance) 

    if len(total_point_to_surface_distance) == 0:
        average_point_to_surface_distance = 0 
    else:
        average_point_to_surface_distance = np.mean(total_point_to_surface_distance) 

    validation_epoch_cd_dist_list.append(average_chamfer_distance)
    validation_epoch_p2s_dist_list.append(average_point_to_surface_distance)


    with open('{0}/{1}/validation_epoch_cd_dist_list.pickle'.format(opt.results_path, opt.name), 'wb') as handle:
    	pickle.dump(validation_epoch_cd_dist_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{0}/{1}/validation_epoch_p2s_dist_list.pickle'.format(opt.results_path, opt.name), 'wb') as handle:
    	pickle.dump(validation_epoch_p2s_dist_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("[Validation] Overall Epoch {0}- Avg CD: {1}; Avg P2S: {2}".format(epoch, average_chamfer_distance, average_point_to_surface_distance ) )


    # Delete files that are created for validation
    for file_path in val_mesh_paths:
        #print('file_path:', file_path)
        mesh_path = file_path
        image_path = file_path.replace('.obj', '.png')
        if os.path.isfile(mesh_path):
            os.remove(mesh_path)

        if os.path.isfile(image_path):
            os.remove(image_path)
        else:
            image_path = image_path[0:-8] + '.png'
            if os.path.isfile(image_path):
                os.remove(image_path)


        except Exception as e:
            print(e)



    plt.plot( np.arange( len(validation_epoch_cd_dist_list) ) , np.array(validation_epoch_cd_dist_list) )
    plt.plot( np.arange( len(validation_epoch_p2s_dist_list) ) , np.array(validation_epoch_p2s_dist_list), '-.' )
    plt.xlabel('Epoch')
    plt.ylabel('Validation Error (CD + P2D)')
    plt.title('Epoch Against Validation Error (CD + P2D)')
    plt.savefig(validation_graph_path)

    try:
        sum_of_cd_and_p2s = np.array(validation_epoch_cd_dist_list) + np.array(validation_epoch_p2s_dist_list)
        sum_of_cd_and_p2s = sum_of_cd_and_p2s.tolist()
        epoch_list = np.arange( len(validation_epoch_cd_dist_list) )
        epoch_list = epoch_list + start_epoch + (epoch_to_start_validation-start_epoch)
        epoch_list = epoch_list.tolist()
        z = zip(epoch_list, validation_epoch_cd_dist_list, validation_epoch_p2s_dist_list, sum_of_cd_and_p2s)
        z = list(z)
        z.sort( key= lambda x:x[3] )
        with open(validation_results_path, 'w') as f:
            for item in z:
                f.write( 'Epoch:{0},  CD:{1},  P2S:{2},  Sum:{3} \n'.format( item[0], item[1], item[2], item[3] )  )


    except:
        print("Unable to save table of validation errors")




def get_calib(param_path):
    param = np.load(param_path, allow_pickle=True)  # param is a np.array that looks similar to a dict.  # ortho_ratio = 0.4 , e.g. scale or y_scale = 0.961994278, e.g. center or vmed = [-1.0486  92.56105  1.0101 ]
    center = param.item().get('center') # is camera 3D center position in the 3D World point space (without any rotation being applied).
    R = param.item().get('R')   # R is used to rotate the CAD model according to a given pitch and yaw.
    scale_factor = param.item().get('scale_factor') # is camera 3D center position in the 3D World point space (without any rotation being applied).
    load_size_associated_with_scale_factor = 1024

    translate = -center.reshape(3, 1)
    extrinsic = np.concatenate([R, translate], axis=1)  # when applied on the 3D pts, the rotation is done first, then the translation
    extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

    scale_intrinsic = np.identity(4)
    scale_intrinsic[0, 0] = 1.0 * scale_factor #2.4851518#1.0   
    scale_intrinsic[1, 1] = -1.0 * scale_factor #-2.4851518#-1.0
    scale_intrinsic[2, 2] = 1.0 * scale_factor  #2.4851518#1.0

    # Match image pixel space to image uv space  (convert a 512x512 image from range of [-256,255] to range of [-1,1] )
    uv_intrinsic = np.identity(4)
    uv_intrinsic[0, 0] = 1.0 / float(load_size_associated_with_scale_factor // 2) # self.opt.loadSizeGlobal == 512 by default. This value must be 512 unless you change the "scale_factor"
    uv_intrinsic[1, 1] = 1.0 / float(load_size_associated_with_scale_factor // 2) # uv_intrinsic[1, 1] is equal to 1/256
    uv_intrinsic[2, 2] = 1.0 / float(load_size_associated_with_scale_factor // 2) 

    intrinsic = np.matmul(uv_intrinsic, scale_intrinsic)
    calib = np.matmul(intrinsic, extrinsic)

    return calib










result_folder = sys.argv[1]
with open('{0}/validation_arguments.pickle'.format(result_folder), 'rb') as handle:
    validation_arguments = pickle.load(handle)



opt = validation_arguments['opt']
val_mesh_paths = validation_arguments['val_mesh_paths']
validation_dataset_frontal_only = validation_arguments['validation_dataset_frontal_only']
num_samples_to_use = validation_arguments['num_samples_to_use']
validation_epoch_cd_dist_list = validation_arguments['validation_epoch_cd_dist_list']
validation_epoch_p2s_dist_list = validation_arguments['validation_epoch_p2s_dist_list']
epoch = validation_arguments['epoch']
validation_graph_path = validation_arguments['validation_graph_path']
start_epoch = validation_arguments['start_epoch']
epoch_to_start_validation = validation_arguments['epoch_to_start_validation']
validation_results_path = validation_arguments['validation_results_path']


validate_meshes(opt, val_mesh_paths, validation_dataset_frontal_only, num_samples_to_use, validation_epoch_cd_dist_list, validation_epoch_p2s_dist_list, epoch, validation_graph_path, start_epoch, epoch_to_start_validation, validation_results_path)
