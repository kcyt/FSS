
import sys
import os
import json
import time 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
print( "torch.cuda.is_available():" , torch.cuda.is_available())
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt

from lib.options import BaseOptions
from lib.model.HGPIFuNetwNML import HGPIFuNetwNML
from lib.data.THuman_dataset import THumanDataset
from lib.mesh_util import save_obj_mesh_with_color, reconstruction
from lib.geometry import index



from multiprocessing import Process, Queue



seed = 10 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


parser = BaseOptions()
opt = parser.parse()
actual_num_epoch = opt.num_epoch
actual_home_dir = opt.home_dir
actual_gpu_id = opt.gpu_id
gen_test_counter = 0

# Start of options check

test_script_activate = False # Set to True to generate the test subject meshes


# Whether to load model weights
load_model_weights = False
load_model_weights_for_high_res_too = False
load_model_weights_for_low_res_finetuning_config = 0 # 0 == No Load weights; 1 == Load, load optimizerG weights; 2 == Load, load optimizer_lowResFineTune weights
checkpoint_folder_to_load_low_res = '{0}/FSS/apps/checkpoints/Date_05_Feb_23_Time_21_55_21 (UpdatedSDFFilter_w_SmplxGuided)'.format(opt.home_dir)
checkpoint_folder_to_load_high_res = '{0}/FSS/apps/checkpoints/Date_06_Feb_23_Time_19_58_56 (HRI w updatedAllThree as Base)'.format(opt.home_dir)
epoch_to_load_from_low_res = 94 
epoch_to_load_from_high_res = 229
start_epoch = 0 # epoch to start training at, usually is 0 or "epoch_to_load_from_low_res+1"


# Whether to load option file
load_option_file = 0 # 0 == No Load options; 1 == Load from low_res_folder; 2 == Load from high_res_folder
load_option_but_different_results_folder = False 

save_model_weights = True

epoch_to_start_validation = 0 # Usually set to 0 or equal to start_epoch. E.g. if start_epoch == 4, and epoch_to_start_validation == 10, then validation will only start on 10

# End of options check






if load_option_file == 0 :
    pass 
else:

    if load_option_file == 1:
        opt_filepath = checkpoint_folder_to_load_low_res.replace('/checkpoints/', '/results/')
    elif load_option_file == 2:
        opt_filepath = checkpoint_folder_to_load_high_res.replace('/checkpoints/', '/results/')
    else:
        raise Exception('load_option_file is wrongly set')

    print('Loading options from {0}'.format(opt_filepath) )

    opt_filepath = os.path.join(opt_filepath, 'opt.txt' )
    with open(opt_filepath, 'r') as f:
        option_vars = json.load(f)

    for k,v in option_vars.items():
        setattr(opt,k,v)


    if load_option_but_different_results_folder:
        # set a new folder for this run
        timestamp = time.strftime('Date_%d_%b_%y_Time_%H_%M_%S')
        opt.name = timestamp
        print("New opt.name is: ", timestamp)
    else:
        if load_option_file == 1:
            opt.name = checkpoint_folder_to_load_low_res.split('/')[-1]
        elif load_option_file == 2:
            opt.name = checkpoint_folder_to_load_high_res.split('/')[-1]
    

    # update the num_epoch
    opt.num_epoch = actual_num_epoch

    # update home dir
    opt.home_dir = actual_home_dir

    # update gpu_id
    opt.gpu_id = actual_gpu_id



gen_test_counter = start_epoch 

generate_point_cloud = True




if opt.use_High_Res_Component:
    opt.sigma_low_resolution_pifu = opt.sigma_high_resolution_pifu
    print("Modifying sigma_low_resolution_pifu to {0} for high resolution component!".format(opt.sigma_high_resolution_pifu) )



def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob >= 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )







def adjust_learning_rate(optimizer, epoch, lr, schedule, learning_rate_decay):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= learning_rate_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr





def gen_mesh(resolution, net, device, data, save_path, thresh=0.5, use_octree=True, generate_from_low_res = False):


    calib_tensor = data['calib'].to(device=device)
    calib_tensor = torch.unsqueeze(calib_tensor,0)
    
    b_min = data['b_min']
    b_max = data['b_max']

    # low-resolution image that is required by both models
    image_low_tensor = data['render_low_pifu'].to(device=device)  
    image_low_tensor = image_low_tensor.unsqueeze(0)

    if opt.use_front_normal:
        nmlF_low_tensor = data['nmlF'].to(device=device)
        nmlF_low_tensor = nmlF_low_tensor.unsqueeze(0)
    else:
        nmlF_low_tensor = None


    if opt.use_back_normal:
        nmlB_low_tensor = data['nmlB'].to(device=device)
        nmlB_low_tensor = nmlB_low_tensor.unsqueeze(0)
    else:
        nmlB_low_tensor = None





    if opt.use_High_Res_Component:
        netG, highRes_netG = net
        net = highRes_netG

        image_high_tensor = data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
        image_high_tensor = torch.unsqueeze(image_high_tensor,0)

        if opt.use_front_normal:
            nmlF_high_tensor = data['nmlF_high_res'].to(device=device)
            nmlF_high_tensor = nmlF_high_tensor.unsqueeze(0)
        else:
            nmlF_high_tensor = None


        if opt.use_back_normal:
            nmlB_high_tensor = data['nmlB_high_res'].to(device=device)
            nmlB_high_tensor = nmlB_high_tensor.unsqueeze(0)
        else:
            nmlB_high_tensor = None
            

        if opt.use_mask_for_rendering_high_res:
            mask_high_res_tensor = data['mask'].to(device=device)
            mask_high_res_tensor = mask_high_res_tensor.unsqueeze(0)
        else:
            mask_high_res_tensor = None




        netG.filter( image_low_tensor, nmlF=nmlF_low_tensor, nmlB = nmlB_low_tensor ) # forward-pass using only the low-resolution PiFU
        netG_output_map = netG.get_im_feat() # should have shape of [B, 256, H, W]

        net.filter( image_high_tensor, nmlF=nmlF_high_tensor, nmlB = nmlB_high_tensor, netG_output_map = netG_output_map , mask_low_res_tensor=None, mask_high_res_tensor=mask_high_res_tensor ) # forward-pass 
        image_tensor = image_high_tensor

    else:

        if opt.use_mask_for_rendering_low_res:
            mask_low_res_tensor = data['mask_low_pifu'].to(device=device)
            mask_low_res_tensor = mask_low_res_tensor.unsqueeze(0)
        else:
            mask_low_res_tensor = None


        net.filter( image_low_tensor, nmlF=nmlF_low_tensor, nmlB = nmlB_low_tensor, netG_output_map = None , mask_low_res_tensor=mask_low_res_tensor, mask_high_res_tensor=None ) # forward-pass 
        image_tensor = image_low_tensor




    

    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)


        verts, faces, _, _ = reconstruction(
            net, device, calib_tensor, resolution, thresh, use_octree=use_octree, num_samples=50000, b_min=b_min , b_max=b_max , generate_from_low_res = generate_from_low_res )


        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=device).float()

        xyz_tensor = net.projection(verts_tensor, calib_tensor) # verts_tensor should have a range of [-1,1]
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor, uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5


        save_obj_mesh_with_color(save_path, verts, faces, color)


    except Exception as e:
        print(e)
        print("Cannot create marching cubes at this time.")






def train(opt):
    global gen_test_counter
    currently_epoch_to_update_low_res_pifu = True
    processes = []
    process_index_to_remove = -1



    if torch.cuda.is_available():
        # set cuda
        device = 'cuda:{0}'.format(str(opt.gpu_id))

    else:
        device = 'cpu'

    print("using device {}".format(device) )





    
    if test_script_activate:
        # Note that we are using the frontal-only test set
        train_dataset = THumanDataset(opt, projection='orthogonal', phase = 'train', evaluation_mode = True, frontal_only=True, frontal_only_but_enforce_multi_angles_still= (gen_test_meshes_from_all_angles or gen_train_val_and_test_meshes_from_all_angles), use_all_subjects = gen_train_val_and_test_meshes_from_all_angles )
            
    else:
        train_dataset = THumanDataset(opt, projection='orthogonal', phase = 'train')
        train_dataset_frontal_only = THumanDataset(opt, projection='orthogonal', phase = 'train', frontal_only=True)


    projection_mode = train_dataset.projection_mode

    if len(train_dataset) < opt.batch_size:
        batch_size = len(train_dataset)
        print("Change batch_size from {0} to {1}".format(opt.batch_size, batch_size) )
    else:
        batch_size = opt.batch_size
        print("Using batch_size == {0}".format(batch_size) )

    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)


    print('train loader size: ', len(train_data_loader))

    if opt.useValidationSet and not test_script_activate :
        validation_dataset_frontal_only = THumanDataset(opt, projection='orthogonal', phase = 'validation', evaluation_mode=False, validation_mode=True, frontal_only=True)

        validation_epoch_cd_dist_list = []
        validation_epoch_p2s_dist_list = []

        validation_graph_path = os.path.join(opt.results_path, opt.name, 'ValidationError_Graph.png')
        validation_results_path = os.path.join(opt.results_path, opt.name, 'validation_results.txt')

    
    netG = HGPIFuNetwNML(opt, projection_mode, use_High_Res_Component = False)

    if opt.use_High_Res_Component:
        highRes_netG = HGPIFuNetwNML(opt, projection_mode, use_High_Res_Component = True)





    if (not os.path.exists(opt.checkpoints_path) ):
        os.makedirs(opt.checkpoints_path)
    if (not os.path.exists(opt.results_path) ):
        os.makedirs(opt.results_path)
    if (not os.path.exists('%s/%s' % (opt.checkpoints_path, opt.name))  ):
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name))
    if (not os.path.exists('%s/%s' % (opt.results_path, opt.name)) ):
        os.makedirs('%s/%s' % (opt.results_path, opt.name))



    if load_model_weights:

        # load weights for low-res model
        modelG_path = os.path.join( checkpoint_folder_to_load_low_res ,"netG_model_state_dict_epoch{0}.pickle".format(epoch_to_load_from_low_res) )

        print('Resuming from ', modelG_path)

        if device == 'cpu' :
            import io

            class CPU_Unpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch.storage' and name == '_load_from_bytes':
                        return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                    else:
                        return super().find_class(module, name)

            with open(modelG_path, 'rb') as handle:
               netG_state_dict = CPU_Unpickler(handle).load()


        else:
            with open(modelG_path, 'rb') as handle:
               netG_state_dict = pickle.load(handle)





        netG.load_state_dict( netG_state_dict , strict = False )
        
        
        
        # load weights for high-res model
        if opt.use_High_Res_Component and load_model_weights_for_high_res_too:
            
            modelhighResG_path = os.path.join( checkpoint_folder_to_load_high_res, "highRes_netG_model_state_dict_epoch{0}.pickle".format(epoch_to_load_from_high_res) )

            print('Resuming from ', modelhighResG_path)

            if device == 'cpu' :
                with open(modelhighResG_path, 'rb') as handle:
                   highResG_state_dict = CPU_Unpickler(handle).load()

            else:
                with open(modelhighResG_path, 'rb') as handle:
                   highResG_state_dict = pickle.load(handle)


            highRes_netG.load_state_dict( highResG_state_dict , strict = False )
            
        
            
            



        
    if test_script_activate:
        # testing script
        with torch.no_grad():

            
            train_dataset.is_train = False
            netG = netG.to(device=device)
            netG.eval()

            if opt.use_High_Res_Component:
                highRes_netG = highRes_netG.to(device=device)
                highRes_netG.eval()


            print('generate mesh (test) ...')

            len_to_iterate = len(train_dataset)
            for gen_idx in tqdm(range(len_to_iterate)):

                index_to_use = gen_idx #gen_test_counter % len(train_dataset)
                #gen_test_counter += 36 #10 # 10 is the number of images for each class
                train_data = train_dataset.get_item(index=index_to_use) 
                # train_data["img"].shape  has shape of [1, 3, 512, 512]

                if gen_train_val_and_test_meshes_from_all_angles:
                    render_path = train_data['render_path']
                    render_path = render_path.split('rendered_image_')[-1]
                    render_angle = render_path.replace('.png','')
                    save_path = '%s/%s/predicted_mesh_subject_%s_angle_%s.obj' % (
                        opt.results_path, opt.name, train_data['name'], render_angle) 
                elif gen_test_meshes_from_all_angles:
                    render_path = train_data['render_path']
                    render_path = render_path.split('rendered_image_')[-1]
                    render_angle = render_path.replace('.png','')
                    save_path = '%s/%s/test_%s_%s.obj' % (
                        opt.results_path, opt.name, train_data['name'], render_angle)
                else:
                    save_path = '%s/%s/test_%s.obj' % (
                        opt.results_path, opt.name, train_data['name'])

                generate_from_low_res = True

                if opt.use_High_Res_Component:
                    gen_mesh(resolution=opt.resolution, net=[netG, highRes_netG] , device = device, data = train_data, save_path = save_path, generate_from_low_res = generate_from_low_res)
                else:
                    gen_mesh(resolution=opt.resolution, net=netG, device = device, data = train_data, save_path = save_path, generate_from_low_res = generate_from_low_res)


            if test_script_generate_for_validation_set:
                print('generate mesh (validation) ...')
                if test_script_activate_option_use_BUFF_dataset:
                    len_to_iterate = len(validation_dataset_frontal_only)
                else:
                    len_to_iterate = len(validation_dataset_frontal_only) #105 #72
                for gen_idx in tqdm(range(len_to_iterate)):

                    if test_script_activate_option_use_BUFF_dataset:
                        index_to_use = gen_idx
                    else:
                        index_to_use = gen_idx #gen_test_counter % len(validation_dataset_frontal_only)
                    #gen_test_counter += 36 #10 # 10 is the number of images for each class
                    val_data = validation_dataset_frontal_only.get_item(index=index_to_use) 
                    # val_data["img"].shape  has shape of [1, 3, 512, 512]
                    save_path = '%s/%s/validation_%s.obj' % (
                        opt.results_path, opt.name, val_data['name'])

                    generate_from_low_res = True

                    if opt.use_High_Res_Component:
                        gen_mesh(resolution=opt.resolution, net=[netG, highRes_netG] , device = device, data = val_data, save_path = save_path, generate_from_low_res = generate_from_low_res)
                    else:
                        gen_mesh(resolution=opt.resolution, net=netG, device = device, data = val_data, save_path = save_path, generate_from_low_res = generate_from_low_res)


        print("Testing is Done! Exiting...")
        return
        
         
    



    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))




    netG = netG.to(device=device)
    lr_G = opt.learning_rate_G
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr_G, momentum=0, weight_decay=0)


    

    if load_model_weights:
        # load saved weights for optimizerG
        optimizerG_path = os.path.join(checkpoint_folder_to_load_low_res, "optimizerG_epoch{0}.pickle".format(epoch_to_load_from_low_res) )

        if device == 'cpu' :
            with open(optimizerG_path, 'rb') as handle:
                optimizerG_state_dict = CPU_Unpickler(handle).load()

        else:
            with open(optimizerG_path, 'rb') as handle:
                optimizerG_state_dict = pickle.load(handle)



        try:
            optimizerG.load_state_dict( optimizerG_state_dict )

        except Exception as e:
            print(e)
            print("Unable to load optimizerG saved weights!")
        
         
    if opt.use_High_Res_Component:
        highRes_netG = highRes_netG.to(device=device)
        lr_highRes = opt.learning_rate_MR
        optimizer_highRes = torch.optim.RMSprop(highRes_netG.parameters(), lr=lr_highRes, momentum=0, weight_decay=0)
        

        if load_model_weights and load_model_weights_for_high_res_too:
            # load highRes optimizer weights
            optimizer_highRes_path = os.path.join(checkpoint_folder_to_load_high_res, "optimizer_highRes_epoch{0}.pickle".format(epoch_to_load_from_high_res) )
            
            if device == 'cpu' :
                with open(optimizer_highRes_path, 'rb') as handle:
                    optimizer_highRes_state_dict = CPU_Unpickler(handle).load()

            else:
                with open(optimizer_highRes_path, 'rb') as handle:
                    optimizer_highRes_state_dict = pickle.load(handle)


            try:
                optimizer_highRes.load_state_dict( optimizer_highRes_state_dict )
            except Exception as e:
                print(e)
                print("Unable to load optimizer_highRes saved weights!")

            
        
        if opt.update_low_res_pifu:
            optimizer_lowResFineTune = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate_low_res_finetune, momentum=0, weight_decay=0)

            if load_model_weights and (load_model_weights_for_low_res_finetuning_config != 0):
                # load optimizer_lowResFineTune weights

                if load_model_weights_for_low_res_finetuning_config == 1:
                    optimizer_lowResFineTune_path = os.path.join(checkpoint_folder_to_load_low_res, "optimizerG_epoch{0}.pickle".format(epoch_to_load_from_low_res) )
                elif load_model_weights_for_low_res_finetuning_config == 2:
                    optimizer_lowResFineTune_path = os.path.join(checkpoint_folder_to_load_high_res, "optimizer_lowResFineTune_epoch{0}.pickle".format(epoch_to_load_from_high_res)  )          
                else:
                    raise Exception('Incorrect use of load_model_weights_for_low_res_finetuning_config!')
                

                if device == 'cpu' :
                    with open(optimizer_lowResFineTune_path, 'rb') as handle:
                        optimizer_lowResFineTune_state_dict = CPU_Unpickler(handle).load()

                else:
                    with open(optimizer_lowResFineTune_path, 'rb') as handle:
                        optimizer_lowResFineTune_state_dict = pickle.load(handle)


                optimizer_lowResFineTune.load_state_dict( optimizer_lowResFineTune_state_dict )
                



    for epoch in range(start_epoch, start_epoch + opt.num_epoch):


        print("start of epoch {}".format(epoch) )

        netG.train()
        if opt.use_High_Res_Component:
            if opt.update_low_res_pifu:
                if (epoch < opt.epoch_to_start_update_low_res_pifu):
                    currently_epoch_to_update_low_res_pifu = False 
                    print("currently_epoch_to_update_low_res_pifu remains at False for this epoch")
                elif (epoch >= opt.epoch_to_end_update_low_res_pifu):
                    currently_epoch_to_update_low_res_pifu = False
                    print("No longer updating low_res_pifu! In the Finetune Phase") 
                elif (epoch % opt.epoch_interval_to_update_low_res_pifu == 0):
                    currently_epoch_to_update_low_res_pifu = not currently_epoch_to_update_low_res_pifu
                    print("Updating currently_epoch_to_update_low_res_pifu to: ",currently_epoch_to_update_low_res_pifu)
                else:
                    pass

            if opt.update_low_res_pifu and currently_epoch_to_update_low_res_pifu:
                netG.train()
                highRes_netG.eval()
            else:
                netG.eval()
                highRes_netG.train()


        epoch_error = 0
        train_len = len(train_data_loader)
        for train_idx, train_data in enumerate(train_data_loader):
            print("batch {}".format(train_idx) )


            # retrieve the data
            calib_tensor = train_data['calib'].to(device=device) # the calibration matrices for the renders ( is np.matmul(intrinsic, extrinsic)  ). Shape of [Batchsize, 4, 4]


            if opt.use_High_Res_Component:
                render_low_pifu_tensor = train_data['render_low_pifu'].to(device=device) 
                render_pifu_tensor = train_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
                
                if opt.use_front_normal:
                    nmlF_low_tensor = train_data['nmlF'].to(device=device)
                    nmlF_tensor = train_data['nmlF_high_res'].to(device=device)
                else:
                    nmlF_low_tensor = None
                    nmlF_tensor = None

                if opt.use_back_normal:
                    nmlB_low_tensor = train_data['nmlB'].to(device=device)
                    nmlB_tensor = train_data['nmlB_high_res'].to(device=device)
                else:
                    nmlB_low_tensor = None
                    nmlB_tensor = None


            else:


                # low-resolution image that is required by both models
                render_pifu_tensor = train_data['render_low_pifu'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
                    
                if opt.use_front_normal:
                    nmlF_tensor = train_data['nmlF'].to(device=device)
                else:
                    nmlF_tensor = None

                if opt.use_back_normal:
                    nmlB_tensor = train_data['nmlB'].to(device=device)
                else:
                    nmlB_tensor = None


            if opt.predict_vertex_normals:
                labels_normal_pred = train_data['all_vertex_normals_labels'].to(device=device).float() 
                if (opt.useDOS and not  opt.useDOS_ButWithSmplxGuide ):
                    vertex_normals_sample_pts = None 
                else:
                    vertex_normals_sample_pts = train_data['vertex_normals_sample_pts'].to(device=device).float()
            else:
                labels_normal_pred = None
                vertex_normals_sample_pts = None




            sdf_plane = None
            if opt.SDF_Filter and opt.SDF_Filter_config == 1 :
                sdf_plane = train_data['sdf_plane'].to(device=device).float() 
 
            # low-resolution pifu
            samples_pifu_tensor = train_data['samples_low_res_pifu'].to(device=device)  # contain inside and outside points. Shape of [Batch_size, 3, num_of_points]
            labels_pifu_tensor = train_data['labels_low_res_pifu'].to(device=device)  # tell us which points in sample_tensor are inside and outside in the surface. Should have shape of [Batch_size ,1, num_of_points]


            if opt.use_High_Res_Component:
                netG.filter( render_low_pifu_tensor, nmlF=nmlF_low_tensor, nmlB = nmlB_low_tensor ) # forward-pass using only the low-resolution PiFU
                netG_output_map = netG.get_im_feat() # should have shape of [B, 256, H, W]

                error_high_pifu, res_high_res_pifu = highRes_netG.forward(images=render_pifu_tensor, points=samples_pifu_tensor, calibs=calib_tensor, labels=labels_pifu_tensor,  points_nml=None, labels_nml=None, nmlF = nmlF_tensor, nmlB = nmlB_tensor, netG_output_map=netG_output_map, labels_normal_pred=labels_normal_pred, vertex_normals_sample_pts=vertex_normals_sample_pts)


                if opt.update_low_res_pifu and currently_epoch_to_update_low_res_pifu:
                    optimizer_lowResFineTune.zero_grad()
                    error_high_pifu['Err(occ)'].backward()
                    curr_high_loss = error_high_pifu['Err(occ)'].item()
                    optimizer_lowResFineTune.step() 
                else:

                    optimizer_highRes.zero_grad()
                    error_high_pifu['Err(occ)'].backward()
                    curr_high_loss = error_high_pifu['Err(occ)'].item()
                    optimizer_highRes.step()

                print(
                'Name: {0} | Epoch: {1} | error_high_pifu: {2:.06f} | LR: {3:.06f} '.format(
                    opt.name, epoch, curr_high_loss, lr_highRes)
                )

                epoch_error += curr_high_loss


            else:

                error_low_res_pifu, res_low_res_pifu = netG.forward(images=render_pifu_tensor, points=samples_pifu_tensor, calibs=calib_tensor, labels=labels_pifu_tensor,  points_nml=None, labels_nml=None, nmlF = nmlF_tensor, nmlB = nmlB_tensor, labels_normal_pred=labels_normal_pred, vertex_normals_sample_pts=vertex_normals_sample_pts, sdf_plane=sdf_plane )
                
                optimizerG.zero_grad()
                error_low_res_pifu['Err(occ)'].backward()
                curr_low_res_loss = error_low_res_pifu['Err(occ)'].item()
                optimizerG.step()


                print(
                'Name: {0} | Epoch: {1} | error_low_res_pifu: {2:.06f} | LR: {3:.06f} '.format(
                    opt.name, epoch, curr_low_res_loss, lr_G)
                )

                epoch_error += curr_low_res_loss





                

        lr_G = adjust_learning_rate(optimizerG, epoch, lr_G, opt.schedule, opt.learning_rate_decay)
        if opt.use_High_Res_Component:
            if opt.update_low_res_pifu and currently_epoch_to_update_low_res_pifu:
                lr_highRes = adjust_learning_rate(optimizer_lowResFineTune, epoch, lr_highRes, opt.schedule, opt.learning_rate_decay)
            else:
                lr_highRes = adjust_learning_rate(optimizer_highRes, epoch, lr_highRes, opt.schedule, opt.learning_rate_decay)


        print("Overall Epoch {0} -  Error for network: {1}".format(epoch, epoch_error/train_len) )


        with torch.no_grad():

            if True:

                if save_model_weights:
                    # save as pickle:
                    with open( '%s/%s/netG_model_state_dict_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                        pickle.dump(netG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                    with open( '%s/%s/optimizerG_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch)) , 'wb') as handle:
                        pickle.dump(optimizerG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)


                if opt.use_High_Res_Component:

                    if generate_point_cloud:
                        r = res_high_res_pifu

                    if save_model_weights:
                        with open( '%s/%s/highRes_netG_model_state_dict_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                            pickle.dump(highRes_netG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                        with open( '%s/%s/optimizer_highRes_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                            pickle.dump(optimizer_highRes.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
                        
                        if opt.use_extremities_module:
                            with open( '%s/%s/optimizer_extremities_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                                pickle.dump(optimizer_extremities.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                        if opt.update_low_res_pifu:
                            with open( '%s/%s/optimizer_lowResFineTune_epoch%s.pickle' % (opt.checkpoints_path, opt.name,str(epoch) ) , 'wb') as handle:
                                pickle.dump(optimizer_lowResFineTune.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                    highRes_netG.eval()
                else:
                    if generate_point_cloud:
                        r = res_low_res_pifu


                print('generate mesh (train) ...')
                train_dataset_frontal_only.is_train = False
                netG.eval()
                for gen_idx in tqdm(range(1)):

                    index_to_use = gen_test_counter % len(train_dataset_frontal_only)
                    gen_test_counter += 1 
                    train_data = train_dataset_frontal_only.get_item(index=index_to_use) 
                    # train_data["img"].shape  has shape of [1, 3, 512, 512]
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])


                    generate_from_low_res = True
                    if opt.use_High_Res_Component:
                        gen_mesh(resolution=opt.resolution, net=[netG, highRes_netG] , device = device, data = train_data, save_path = save_path, generate_from_low_res = generate_from_low_res)
                    else:
                        gen_mesh(resolution=opt.resolution, net=netG , device = device, data = train_data, save_path = save_path, generate_from_low_res = generate_from_low_res)

                if generate_point_cloud:
                    try:
                        # save visualization of model performance
                        save_path = '%s/%s/pred.ply' % (opt.results_path, opt.name)
                        r = r[0].cpu() # get only the first example in the batch (i.e. 1 CAD model or subject). [1, Num of sampled points]
                        points = samples_pifu_tensor[0].transpose(0, 1).cpu()    # note that similar to res[0], we only take sample_tensor[0] i.e. the first CAD model. Shape of [Num of sampled points, 3] after the transpose. 
                        save_samples_truncted_prob(save_path, points.detach().numpy(), r.detach().numpy())
                    except:
                        print("Unable to save point cloud.")
                    
                train_dataset_frontal_only.is_train = True



            if opt.useValidationSet and (epoch >= epoch_to_start_validation) :
                import trimesh
                from evaluation_helper_functions import quick_get_chamfer_and_surface_dist

                num_samples_to_use = 4000

                print('Commencing validation..')
                print('generate mesh (validation) ...')


                for j, proc in enumerate(processes):

                    # if there is outstanding processes, wait for it first.
                    if proc.poll() is None:
                        print("Previous Validation not finished. Waiting..")
                        proc.wait()
                        #while proc.poll() is None:
                        #    time.sleep(1.0)
                    print("Previous Validation already done")
                    process_index_to_remove = j

                    with open('{0}/{1}/validation_epoch_cd_dist_list.pickle'.format(opt.results_path, opt.name), 'rb') as handle:
                        validation_epoch_cd_dist_list = pickle.load(handle)

                    with open('{0}/{1}/validation_epoch_p2s_dist_list.pickle'.format(opt.results_path, opt.name), 'rb') as handle:
                        validation_epoch_p2s_dist_list = pickle.load(handle)

                if process_index_to_remove != -1:
                    processes.pop(process_index_to_remove)
                    process_index_to_remove = -1

                netG.eval()
                if opt.use_High_Res_Component:
                    highRes_netG.eval()
                val_len = len(validation_dataset_frontal_only)
                val_mesh_paths = []
                index_to_use_list = []
                #num_of_val_examples_to_try = 10
                num_of_val_examples_to_try = val_len
                for gen_idx in tqdm(range(num_of_val_examples_to_try)):
                    print('[Validation] generating mesh #{0}'.format(gen_idx) )

                    index_to_use = gen_idx
                    """
                    index_to_use = np.random.randint(low=0, high=val_len)
                    while index_to_use in index_to_use_list:
                        print('repeated index_to_use is selected, re-sampling')
                        index_to_use = np.random.randint(low=0, high=val_len)
                    index_to_use_list.append(index_to_use)
                    """
                    val_data = validation_dataset_frontal_only.get_item(index=index_to_use) 

                    save_path = '%s/%s/val_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, val_data['name'])

                    val_mesh_paths.append(save_path)

                    generate_from_low_res = True
                    if opt.use_High_Res_Component:
                        gen_mesh(resolution=opt.resolution, net=[netG, highRes_netG] , device = device, data = val_data, save_path = save_path, generate_from_low_res = generate_from_low_res)
                    else:
                        gen_mesh(resolution=opt.resolution, net=netG , device = device, data = val_data, save_path = save_path, generate_from_low_res = generate_from_low_res)


                import subprocess
                from subprocess import Popen
                with open('{0}/{1}/validation_arguments.pickle'.format(opt.results_path,opt.name), 'wb') as handle:
                    validation_arguments = {"opt":opt, "val_mesh_paths":val_mesh_paths, "validation_dataset_frontal_only":validation_dataset_frontal_only, "num_samples_to_use":num_samples_to_use, "validation_epoch_cd_dist_list":validation_epoch_cd_dist_list, "validation_epoch_p2s_dist_list":validation_epoch_p2s_dist_list, "epoch":epoch, "validation_graph_path":validation_graph_path, "start_epoch":start_epoch, "epoch_to_start_validation":epoch_to_start_validation, "validation_results_path":validation_results_path }
                    pickle.dump(validation_arguments, handle, protocol=pickle.HIGHEST_PROTOCOL)
                current_folder = "{0}/{1}".format(opt.results_path,opt.name) 
                validation_process = Popen(
                    ['nohup' , 'python' , 'validation_process.py', current_folder ],
                    stdout = open('{0}/{1}/validation_logfile.txt'.format(opt.results_path,opt.name) , 'a' ),
                    stderr = open('{0}/{1}/debugging_error_logfile.txt'.format(opt.results_path,opt.name), 'a'),
                    start_new_session=True )
                print("Validation Process created!")

                processes.append(validation_process)




    # if there is outstanding processes, wait for it first.
    for p in processes:
        print("Checking for outstanding processes before exiting..")
        p.wait()
    print("No outstanding processes detected... Exiting!")

    unwanted_file = '{0}/{1}/validation_arguments.pickle'.format(opt.results_path,opt.name)
    if os.path.isfile(unwanted_file):
        os.remove(unwanted_file)
    print('Done with training!')







if __name__ == '__main__':
    train(opt)
