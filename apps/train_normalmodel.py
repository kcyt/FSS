
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt


from PIL import Image

from lib.options import BaseOptions
from lib.networks import define_G
from lib.data.THuman_NormalDataset import NormalDataset

import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn


seed = 10 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0

lr = 2e-4  
normal_schedule = [999] # epoch to reduce lr at
print("changing batch size for normal model!")
opt.batch_size = 2
opt.num_epoch = 70


use_VGG_loss = False




def adjust_learning_rate(optimizer_list, epoch, lr, schedule, learning_rate_decay):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= learning_rate_decay
        for optimizer in optimizer_list:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    return lr





class VGG19(nn.Module):
    ''' Wrapper for pretrained torchvision.models.vgg19 to output intermediate feature maps '''

    def __init__(self):
        super().__init__()

        vgg_features = models.vgg19(pretrained=True).features

        self.f1 = nn.Sequential(*[vgg_features[x] for x in range(2)])
        self.f2 = nn.Sequential(*[vgg_features[x] for x in range(2, 7)])
        self.f3 = nn.Sequential(*[vgg_features[x] for x in range(7, 12)])
        self.f4 = nn.Sequential(*[vgg_features[x] for x in range(12, 21)])
        self.f5 = nn.Sequential(*[vgg_features[x] for x in range(21, 30)])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h1 = self.f1(x)
        h2 = self.f2(h1)
        h3 = self.f3(h2)
        h4 = self.f4(h3)
        h5 = self.f5(h4)
        return [h1, h2, h3, h4, h5]


def vgg_loss(vgg, x_real, x_fake, vgg_weights):
        ''' Computes perceptual loss with VGG network from real and fake images '''
        vgg_real = vgg(x_real)
        vgg_fake = vgg(x_fake)

        vgg_loss = 0.0
        for real, fake, weight in zip(vgg_real, vgg_fake, vgg_weights):
            vgg_loss += weight * F.l1_loss(real.detach(), fake)
        return vgg_loss




def train(opt):
    global gen_test_counter
    global lr 

    if torch.cuda.is_available():
        # set cuda
        device = 'cuda:0'

    else:
        device = 'cpu'

    print("using device {}".format(device) )
    
    train_dataset = NormalDataset(opt, evaluation_mode=False)

    train_dataset_frontal_only = NormalDataset(opt, evaluation_mode=False, frontal_only=True)

    

    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)


    print('train loader size: ', len(train_data_loader))

    if opt.useValidationSet:
        validation_dataset = NormalDataset(opt, phase = 'validation', evaluation_mode=False, validation_mode=True)
        validation_data_loader = DataLoader(validation_dataset, 
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)
        validation_epoch_error_netF_list = []
        validation_epoch_error_netB_list = []

        validation_graph_path = os.path.join(opt.results_path, opt.name, 'ValidationLoss_Graph.png')


    if use_VGG_loss:
        vgg = VGG19()
        vgg_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    smoothL1Loss = nn.SmoothL1Loss()


    
    netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")

    netB = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")



    """
    # Load from saved model weights
    F_modelnormal_path = "/mnt/lustre/kennard.chan/specialized_pifuhd/apps/checkpoints/Date_12_Nov_21_Time_01_38_54/netF_model_state_dict.pickle"
    B_modelnormal_path = "/mnt/lustre/kennard.chan/specialized_pifuhd/apps/checkpoints/Date_12_Nov_21_Time_01_38_54/netB_model_state_dict.pickle"

    print('Resuming from ', F_modelnormal_path)
    print('Resuming from ', B_modelnormal_path)

    with open(F_modelnormal_path, 'rb') as handle:
       netF_state_dict = pickle.load(handle)

    with open(B_modelnormal_path, 'rb') as handle:
       netB_state_dict = pickle.load(handle)

    netF.load_state_dict( netF_state_dict , strict = True )
    netB.load_state_dict( netB_state_dict , strict = True )
    """
    
        



    if (not os.path.exists(opt.checkpoints_path) ):
        os.makedirs(opt.checkpoints_path)
    if (not os.path.exists(opt.results_path) ):
        os.makedirs(opt.results_path)
    if (not os.path.exists('%s/%s' % (opt.checkpoints_path, opt.name))  ):
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name))
    if (not os.path.exists('%s/%s' % (opt.results_path, opt.name)) ):
        os.makedirs('%s/%s' % (opt.results_path, opt.name))



    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))




    netF = netF.to(device=device)
    netB = netB.to(device=device)

    if use_VGG_loss:
        vgg = vgg.to(device=device)
    
 
    optimizer_netF = torch.optim.RMSprop(netF.parameters(), lr=lr, momentum=0, weight_decay=0)
    optimizer_netB = torch.optim.RMSprop(netB.parameters(), lr=lr, momentum=0, weight_decay=0)



    start_epoch = 0
    for epoch in range(start_epoch, opt.num_epoch):

        print("start of epoch {}".format(epoch) )

        netF.train()
        netB.train()

        epoch_error_netF = 0
        epoch_error_netB = 0
        train_len = len(train_data_loader)
        for train_idx, train_data in enumerate(train_data_loader):
            print("batch {}".format(train_idx) )

            # retrieve the data
            render_tensor = train_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
            nmlF_high_res_tensor = train_data['nmlF_high_res'].to(device=device)   # shape of [batch, 3,1024,1024]
            nmlB_high_res_tensor = train_data['nmlB_high_res'].to(device=device)   # shape of [batch, 3,1024,1024]

        
            res_netF = netF.forward(render_tensor)
            res_netB = netB.forward(render_tensor)

            if use_VGG_loss:
                err_netF = (
                5.0 * smoothL1Loss(res_netF, nmlF_high_res_tensor) + \
                1.0 * vgg_loss(vgg=vgg, x_real=nmlF_high_res_tensor, x_fake=res_netF, vgg_weights=vgg_weights)
                )

                err_netB = (
                5.0 * smoothL1Loss(res_netB, nmlB_high_res_tensor) + \
                1.0 * vgg_loss(vgg=vgg, x_real=nmlB_high_res_tensor, x_fake=res_netB, vgg_weights=vgg_weights)
                )

            else:
                err_netF = smoothL1Loss(res_netF, nmlF_high_res_tensor) 
                err_netB = smoothL1Loss(res_netB, nmlB_high_res_tensor)
       


        
            optimizer_netF.zero_grad()
            err_netF.backward()
            curr_loss_netF = err_netF.item()
            optimizer_netF.step()

            optimizer_netB.zero_grad()
            err_netB.backward()
            curr_loss_netB = err_netB.item()
            optimizer_netB.step()

            print(
            'Name: {0} | Epoch: {1} | curr_loss_netF: {2:.06f} | curr_loss_netB: {3:.06f}  | LR: {4:.06f} '.format(
                opt.name, epoch, curr_loss_netF, curr_loss_netB, lr)
            )

            epoch_error_netF += curr_loss_netF 
            epoch_error_netB += curr_loss_netB

                


        lr = adjust_learning_rate( [optimizer_netF, optimizer_netB] , epoch, lr, schedule=normal_schedule, learning_rate_decay=0.1)
        print("Overall Epoch {0} -  Error for netF: {1};   Error for netB: {2}".format(epoch, epoch_error_netF/train_len, epoch_error_netB/train_len) )



        with torch.no_grad():
            #if (  ( epoch%10==0) or epoch==opt.num_epoch-1   ):
            if True:


                # save models
                with open( '%s/%s/netF_model_state_dict_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                    pickle.dump(netF.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open( '%s/%s/netB_model_state_dict_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                    pickle.dump(netB.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                print('generate normal map (train) ...')
                train_dataset_frontal_only.is_train = False
                netF.eval()
                netB.eval()
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):

                    index_to_use = gen_test_counter % len(train_dataset_frontal_only)
                    gen_test_counter += 1 
                    train_data = train_dataset_frontal_only.get_item(index=index_to_use) 
                    # train_data["img"].shape  has shape of [1, 3, 512, 512]
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])


                    image_tensor = train_data['original_high_res_render'].to(device=device)  # 512 x 512  
                    image_tensor = torch.unsqueeze(image_tensor,0)
        
                    #image_tensor = torch.cat( [image_tensor ], dim=1 )


                    original_nmlF_map = train_data['nmlF_high_res'].cpu().numpy()
                    original_nmlB_map = train_data['nmlB_high_res'].cpu().numpy()



                    res_netF = netF.forward(image_tensor)
                    res_netB = netB.forward(image_tensor)

                    res_netF = res_netF.detach().cpu().numpy()[0,:,:,:]
                    res_netB = res_netB.detach().cpu().numpy()[0,:,:,:]


                    save_netF_normalmap_path = save_path[:-4] + 'netF_normalmap.png'
                    save_netB_normalmap_path = save_path[:-4] + 'netB_normalmap.png'
                    numpy_save_netF_normalmap_path = save_path[:-4] + 'netF_normalmap.npy'
                    numpy_save_netB_normalmap_path = save_path[:-4] + 'netB_normalmap.npy'
                    GT_netF_normalmap_path = save_path[:-4] + 'netF_groundtruth.png'
                    GT_netB_normalmap_path = save_path[:-4] + 'netB_groundtruth.png'

                    np.save(numpy_save_netF_normalmap_path , res_netF)
                    np.save(numpy_save_netB_normalmap_path , res_netB)


                    save_netF_normalmap = (np.transpose(res_netF, (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_netF_normalmap = save_netF_normalmap.astype(np.uint8)
                    save_netF_normalmap = Image.fromarray(save_netF_normalmap)
                    save_netF_normalmap.save(save_netF_normalmap_path)

                    save_netB_normalmap = (np.transpose(res_netB, (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_netB_normalmap = save_netB_normalmap.astype(np.uint8)
                    save_netB_normalmap = Image.fromarray(save_netB_normalmap)
                    save_netB_normalmap.save(save_netB_normalmap_path)


                    GT_netF_normalmap = (np.transpose(original_nmlF_map, (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    GT_netF_normalmap = GT_netF_normalmap.astype(np.uint8)
                    GT_netF_normalmap = Image.fromarray(GT_netF_normalmap)
                    GT_netF_normalmap.save(GT_netF_normalmap_path)

                    GT_netB_normalmap = (np.transpose(original_nmlB_map, (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    GT_netB_normalmap = GT_netB_normalmap.astype(np.uint8)
                    GT_netB_normalmap = Image.fromarray(GT_netB_normalmap)
                    GT_netB_normalmap.save(GT_netB_normalmap_path)

                if opt.useValidationSet:
                    print('Commencing validation..')
                    validation_epoch_error_netF = 0
                    validation_epoch_error_netB = 0
                    val_len = len(validation_data_loader)
                    for val_idx, val_data in enumerate(validation_data_loader):
                        print("val batch {}".format(val_idx) )

                        # retrieve the data
                        render_tensor = val_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
                        nmlF_high_res_tensor = val_data['nmlF_high_res'].to(device=device)   # shape of [batch, 3,1024,1024]
                        nmlB_high_res_tensor = val_data['nmlB_high_res'].to(device=device)   # shape of [batch, 3,1024,1024]

                    
                        res_netF = netF.forward(render_tensor)
                        res_netB = netB.forward(render_tensor)

                        if use_VGG_loss:
                            err_netF = (
                            5.0 * smoothL1Loss(res_netF, nmlF_high_res_tensor) + \
                            1.0 * vgg_loss(vgg=vgg, x_real=nmlF_high_res_tensor, x_fake=res_netF, vgg_weights=vgg_weights)
                            )

                            err_netB = (
                            5.0 * smoothL1Loss(res_netB, nmlB_high_res_tensor) + \
                            1.0 * vgg_loss(vgg=vgg, x_real=nmlB_high_res_tensor, x_fake=res_netB, vgg_weights=vgg_weights)
                            )

                        else:
                            err_netF = smoothL1Loss(res_netF, nmlF_high_res_tensor) 
                            err_netB = smoothL1Loss(res_netB, nmlB_high_res_tensor)
                   

                        curr_loss_netF = err_netF.item()
                        curr_loss_netB = err_netB.item()


                        print(
                        '[Validation] Name: {0} | Epoch: {1} | curr_loss_netF: {2:.06f} | curr_loss_netB: {3:.06f}  | LR: {4:.06f} '.format(
                            opt.name, epoch, curr_loss_netF, curr_loss_netB, lr)
                        )

                        validation_epoch_error_netF += curr_loss_netF 
                        validation_epoch_error_netB += curr_loss_netB




                    validation_epoch_error_netF_list.append(validation_epoch_error_netF/val_len)     
                    validation_epoch_error_netB_list.append(validation_epoch_error_netB/val_len)  
                    print("[Validation] Overall Epoch {0}-  netF Error: {1} ; netB Error: {2}".format(epoch, validation_epoch_error_netF/val_len, validation_epoch_error_netB/val_len) )




                train_dataset_frontal_only.is_train = True

        plt.plot( np.arange(epoch+1) , np.array(validation_epoch_error_netF_list) )
        plt.plot( np.arange(epoch+1) , np.array(validation_epoch_error_netB_list), '-.' )
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Epoch Against Validation Loss')
        plt.savefig(validation_graph_path)




if __name__ == '__main__':
    train(opt)

