# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import HGFilter

from ..net_util import init_net
from ..net_util import CustomBCELoss
from ..networks import define_G
import cv2
from math import radians


class HGPIFuNetwNML(BasePIFuNet):
    '''
    HGPIFu uses stacked hourglass as an image encoder.
    '''

    def __init__(self, 
                 opt, 
                 projection_mode='orthogonal',
                 criteria={'occ': nn.MSELoss()},
                 use_High_Res_Component = False,
                 ):
        super(HGPIFuNetwNML, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria)

        self.name = 'hg_pifu_low_res'

        self.opt = opt

        self.use_High_Res_Component = use_High_Res_Component

        in_ch = 3
        high_res_in_ch = 3

        try:
            if opt.use_front_normal: # by default is False
                in_ch += 3
                high_res_in_ch += 3
            if opt.use_back_normal: # by default is False
                in_ch += 3
                high_res_in_ch += 3
        except:
            pass




        if self.use_High_Res_Component:
            from .DifferenceIntegratedHGFilters import DifferenceIntegratedHGFilter
            self.image_filter = DifferenceIntegratedHGFilter(1, 2, high_res_in_ch, 256,   # opt.hg_dim will be the channel dimension of the final output generated by the HGFilter
                                        opt.hg_down, False, High_Res_Component_Config=opt.High_Res_Component_Config) 


        else:

            if opt.SDF_Filter:

                if (opt.SDF_Filter_config == 1) or (opt.SDF_Filter_config == 0):
                    # just use sigmoid
                    self.image_filter = HGFilter(opt.num_stack_low_res, opt.hg_depth_low_res, in_ch, opt.hg_dim_low_res,   # opt.hg_dim will be the channel dimension of the final output generated by the HGFilter
                                 opt.norm, opt.hg_down, True, SDF_Filter_subconfig_except_last=opt.SDF_Filter_subconfig_except_last, SDF_Filter_subconfig_except_penultimate=opt.SDF_Filter_subconfig_except_penultimate ) 
                else:
                    raise Exception("opt.SDF_Filter_config is wrongly set!")

            
            else:
                # below, opt.num_stack = 4, hg_depth = 2, in_ch = 9 (if opt.use_front_normal and opt.use_back_normal are both set to True )
                # opt.hg_dim = 256, opt.norm = "batch", opt.hg_down = ave_pool by default
                self.image_filter = HGFilter(opt.num_stack_low_res, opt.hg_depth_low_res, in_ch, opt.hg_dim_low_res,   # opt.hg_dim will be the channel dimension of the final output generated by the HGFilter
                                             opt.norm, opt.hg_down, False ) 


       
        mlp_dim_low_res = self.opt.mlp_dim_low_res.copy()




        if opt.useDOS and opt.useDOS_ButWithSmplxGuide :
            _last_op = modified_Tanh(factor=1.5, translation=0.5)
        else:
            _last_op = nn.Sigmoid()

        self.mlp = MLP(
            filter_channels=mlp_dim_low_res, # opt.mlp_dim is defaulted to [257, 1024, 512, 256, 128, 1]
            merge_layer=self.opt.merge_layer_low_res, # opt.merge_layer is defaulted to -1
            res_layers=self.opt.mlp_res_layers_low_res,  # opt.mlp_res_layers is defaulted to [2,3,4]
            #norm=self.opt.mlp_norm,  # opt.mlp_norm is defaulted to "group"
            norm="no_norm",
            last_op=_last_op )

        if self.opt.predict_vertex_normals and (not self.use_High_Res_Component):
            self.criteria['normal_pred'] = nn.MSELoss()

            self.mlp_normal_pred = MLP(
                filter_channels=self.opt.mlp_dim_normal_pred, # opt.mlp_dim is defaulted to [257, 1024, 512, 256, 128, 1]
                merge_layer=self.opt.merge_layer_low_res, 
                res_layers=self.opt.mlp_res_layers_low_res,  # opt.mlp_res_layers is defaulted to [2,3,4]
                #norm=self.opt.mlp_norm,  # opt.mlp_norm is defaulted to "group"
                norm="no_norm",
                last_op=nn.Tanh())


        else: 
            self.mlp_normal_pred = None



        if (not self.use_High_Res_Component) and self.opt.SDF_Filter:
            if self.opt.SDF_Filter_config == 1:
                self.criteria['sdf_plane'] = nn.MSELoss()



        self.spatial_enc = DepthNormalizer(opt, low_res_pifu = True)

        self.im_feat_list = []
        self.tmpx = None
        self.normx = None
        self.phi = None

        self.intermediate_preds_list = []

        self.intermediate_preds_list_normal_pred = [] 
        self.preds_normal_pred = None


        init_net(self) # initialise the weights of the network (If multiple gpu(s) is to be used, then the arguments to this function can be changed)

        self.netF = None
        self.netB = None


        self.nmlF = None
        self.nmlB = None

        self.intermediate_sdf_slices = None




    def filter(self, images, nmlF=None, nmlB = None, netG_output_map = None, mask_low_res_tensor=None, mask_high_res_tensor=None, sdf_plane=None ):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B, C, H, W]
        '''

        self.mask_high_res_tensor = mask_high_res_tensor
        self.mask_low_res_tensor = mask_low_res_tensor

        self.sdf_plane = sdf_plane



        nmls = []
        # if you wish to train jointly, remove detach etc.
        with torch.no_grad():
            #if self.netF is not None:
            if self.opt.use_front_normal:
                if nmlF == None:
                    raise Exception("NORMAL MAPS ARE MISSING!!")

                #self.nmlF = self.netF.forward(images).detach() # frontal normal map 
                self.nmlF = nmlF
                nmls.append(self.nmlF)
            #if self.netB is not None:
            if self.opt.use_back_normal:
                if nmlB == None:
                    raise Exception("NORMAL MAPS ARE MISSING!!")

                #self.nmlB = self.netB.forward(images).detach() # backside normal map
                self.nmlB = nmlB
                nmls.append(self.nmlB)
        
        

        # Concatenate the input image with the two normals maps together
        if len(nmls) != 0:
            nmls = torch.cat(nmls,1)
            if images.size()[2:] != nmls.size()[2:]:
                nmls = nn.Upsample(size=images.size()[2:], mode='bilinear', align_corners=True)(nmls)
            images = torch.cat([images,nmls],1)




        # self.normx is not really used
        if self.use_High_Res_Component: 
            self.im_feat_list, self.normx = self.image_filter(images, netG_output_map) 
        else:
            self.im_feat_list, self.normx = self.image_filter(images) # a list of [B,256,128,128] or [B, C, H, W] 

        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
        
    def query(self, points, calibs, transforms=None, labels=None, update_pred=True, update_phi=True, labels_normal_pred=None, vertex_normals_sample_pts=None ):
        '''
        given 3d points, we obtain 2d projection of these given the camera matrices.
        filter needs to be called beforehand.
        the prediction is stored to self.preds
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image. If calibs is [B,3,4], it is fine as well.
            transforms: [B, 2, 3] image space coordinate transforms
            labels: [B, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction
        '''
        xyz = self.projection(points, calibs, transforms) # [B, 3, N]
        xy = xyz[:, :2, :] # [B, 2, N]

        if self.opt.predict_vertex_normals and (not self.use_High_Res_Component) and not (self.opt.useDOS and not self.opt.useDOS_ButWithSmplxGuide ) and labels_normal_pred is not None:
            xyz_vertex_normals = self.projection(vertex_normals_sample_pts, calibs, transforms) # [B,3,N]
            xy_vertex_normals = xyz_vertex_normals[:, :2, :] # [B, 2, N]
            self.labels_normal_pred = labels_normal_pred




        if self.use_High_Res_Component and self.opt.use_mask_for_rendering_high_res and (self.mask_high_res_tensor is not None):
            mask_values = self.index(self.mask_high_res_tensor , xy, mode='nearest' ) 
        if (not self.use_High_Res_Component) and self.opt.use_mask_for_rendering_low_res and (self.mask_low_res_tensor is not None):
            mask_values = self.index(self.mask_low_res_tensor , xy, mode='nearest'  ) 

        # if the point is outside bounding box, return outside.
        in_bb = (xyz >= -1) & (xyz <= 1) # [B, 3, N]
        in_bb = in_bb[:, 0, :] & in_bb[:, 1, :] & in_bb[:, 2, :] # [B, N]
        in_bb = in_bb[:, None, :].detach().float() # [B, 1, N]

        is_zero_bool = (xyz == 0) # [B, 3, N]; remove the (0,0,0) point that has been used to discard unwanted sample pts
        is_zero_bool = is_zero_bool[:, 0, :] & is_zero_bool[:, 1, :] & is_zero_bool[:, 2, :] # [B, N]
        not_zero_bool = torch.logical_not(is_zero_bool)
        not_zero_bool = not_zero_bool[:, None, :].detach().float() # [B, 1, N]

        if labels is not None:
            self.labels = in_bb * labels # [B, 1, N]
            self.labels = not_zero_bool * self.labels

            if (not self.use_High_Res_Component) and self.opt.SDF_Filter:
                self.labels_truncated = self.labels.clone().detach() 
                self.labels_truncated[self.labels_truncated>1.0] = 1.0  
                self.labels_truncated[self.labels_truncated<0.0] = 0.0


  

        if self.opt.predict_vertex_normals and (not self.use_High_Res_Component) and labels_normal_pred is not None and (self.opt.useDOS and not self.opt.useDOS_ButWithSmplxGuide ):
            self.labels_normal_pred = in_bb * labels_normal_pred # [B, 3, N]
            self.labels_normal_pred = not_zero_bool * self.labels_normal_pred

        sp_feat = self.spatial_enc(xyz, calibs=calibs) # sp_feat is the normalized z value. (x and y are removed)
        
        if self.opt.predict_vertex_normals and (not self.use_High_Res_Component) and labels_normal_pred is not None and not (self.opt.useDOS and not self.opt.useDOS_ButWithSmplxGuide ):
            sp_feat_vertex_normals = self.spatial_enc(xyz_vertex_normals, calibs=calibs) # sp_feat is the normalized z value. (x and y are removed)





        intermediate_preds_list = []
        intermediate_preds_list_normal_pred = [] 

        phi = None
        for i, im_feat in enumerate(self.im_feat_list):

            # im_feat has shape of [B, C, H, W] ; labels_positional_encoding has shape of [Batch, NUM_OF_FACES*6, 128, 128 ]
            if self.opt.predict_vertex_normals and (not self.use_High_Res_Component) and labels_normal_pred is not None:

                if ( self.opt.pred_vertex_normals_only_for_last_layer and (i != ( len(self.im_feat_list) -1 )  )  ) and (self.opt.pred_vertex_normals_only_for_penultimate and (i != ( len(self.im_feat_list) -2 )  ))  :
                    pass 
                else:

                    if not (self.opt.useDOS and not self.opt.useDOS_ButWithSmplxGuide  ):
                        point_local_feat_list_vertex_normal = [self.index(im_feat, xy_vertex_normals ), sp_feat_vertex_normals] # z_feat has already gone through a round of indexing. 'point_local_feat_list' should have shape of [batch_size, 272, num_of_points]     
                        point_local_feat_vertex_normal = torch.cat(point_local_feat_list_vertex_normal, 1)
                        pred_normal_pred, phi_normal_pred = self.mlp_normal_pred(point_local_feat_vertex_normal) # pred_normal_pred has sha[e [B, 3, N]
                    else:
                        pred_normal_pred, phi_normal_pred = self.mlp_normal_pred(point_local_feat)
                        pred_normal_pred = in_bb * pred_normal_pred
                        pred_normal_pred = not_zero_bool * pred_normal_pred
                        if self.use_High_Res_Component and self.opt.use_mask_for_rendering_high_res and (self.mask_high_res_tensor is not None):
                            pred_normal_pred = mask_values * pred_normal_pred
                        if (not self.use_High_Res_Component) and self.opt.use_mask_for_rendering_low_res and (self.mask_low_res_tensor is not None):
                            pred_normal_pred = mask_values * pred_normal_pred

                    intermediate_preds_list_normal_pred.append(pred_normal_pred)



            if (not self.use_High_Res_Component) and self.opt.SDF_Filter and not (self.opt.SDF_Filter_subconfig_except_last and ( i == (len(self.im_feat_list)-1)  ) ) and not ( self.opt.SDF_Filter_subconfig_except_penultimate and ( i == (len(self.im_feat_list)-2)  )  )  :

                uv = xyz
                uv = uv.transpose(1, 2) # [B, N, 3]
                uv = uv[:, :, None, None, :] # [B, N, 1, 1, 3]

                pred = torch.nn.functional.grid_sample(im_feat[:, None, :, : ,:], uv, mode='bilinear' ) #[B, C==1, N, 1, 1]
                pred = pred[:,:,:,0,0] #[B, C==1, N ]
                phi = None
            else:

                point_local_feat_list = [self.index(im_feat, xy ), sp_feat] # z_feat has already gone through a round of indexing.   
                point_local_feat = torch.cat(point_local_feat_list, 1) # 'point_local_feat' should have shape of [batch_size, 256+1, num_of_points]   
                pred, phi = self.mlp(point_local_feat) # phi is activations from an intermediate layer of the MLP. 'pred' has shape of [batch_size, 1, num_of_points]   
            pred = in_bb * pred
            pred = not_zero_bool * pred
            if self.use_High_Res_Component and self.opt.use_mask_for_rendering_high_res and (self.mask_high_res_tensor is not None):
                pred = mask_values * pred
            if (not self.use_High_Res_Component) and self.opt.use_mask_for_rendering_low_res and (self.mask_low_res_tensor is not None):
                pred = mask_values * pred

            intermediate_preds_list.append(pred)



        
        if update_phi:
            self.phi = phi

        if update_pred:
            self.intermediate_preds_list = intermediate_preds_list
            self.preds = self.intermediate_preds_list[-1]

            if self.opt.predict_vertex_normals and (not self.use_High_Res_Component) and labels_normal_pred is not None:
                self.intermediate_preds_list_normal_pred = intermediate_preds_list_normal_pred
                self.preds_normal_pred = self.intermediate_preds_list_normal_pred[-1]



    def get_im_feat(self):
        '''
        return the image filter in the last stack
        return:
            [B, C, H, W]
        '''
        return self.im_feat_list[-1]


    def get_error(self,points=None):
        '''
        return the loss given the ground truth labels and prediction
        '''
        error = {}

        error['Err(occ)'] = 0
        for i,preds in enumerate(self.intermediate_preds_list):

            if (not self.use_High_Res_Component) and self.opt.SDF_Filter and self.opt.SDF_Filter_config==1 and (self.opt.useDOS and  self.opt.useDOS_ButWithSmplxGuide   ):
                if ( self.opt.SDF_Filter_subconfig_except_last and ( i == (len(self.im_feat_list)-1 ) ) ) or ( self.opt.SDF_Filter_subconfig_except_penultimate and ( i == (len(self.im_feat_list)-2 ) ) ) : 
                    labels_to_use = self.labels
                else:
                    labels_to_use = self.labels_truncated
            else:
                labels_to_use = self.labels

            error['Err(occ)'] += self.criteria['occ'](preds, labels_to_use)


            if (not self.use_High_Res_Component) and self.opt.SDF_Filter and self.opt.SDF_Filter_config==1 and self.opt.SDF_Filter_subconfig_except_penultimate and ( i == (len(self.im_feat_list)-2 ) ): 
                error['Err(occ)'] += 1.0 * self.criteria['occ'](preds, labels_to_use) # additional weightage for the penultimate stack because it does not have the 'Err(sdf_plane)'
        

            if (not self.use_High_Res_Component) and self.opt.SDF_Filter and self.opt.SDF_Filter_config==1 and self.opt.SDF_Filter_subconfig_except_last and ( i == (len(self.im_feat_list)-1 ) ): 
                error['Err(occ)'] += 1.0 * self.criteria['occ'](preds, labels_to_use) # additional weightage for the last stack because it does not have the 'Err(sdf_plane)'
        
        error['Err(occ)'] /= len(self.intermediate_preds_list)




        if self.opt.predict_vertex_normals and (not self.use_High_Res_Component):
            error['Err(normal_pred)'] = 0
            for preds_normal_pred in self.intermediate_preds_list_normal_pred:
                error['Err(normal_pred)'] += self.criteria['normal_pred'](preds_normal_pred, self.labels_normal_pred)
            
            error['Err(normal_pred)'] /= len(self.intermediate_preds_list_normal_pred)




        if (not self.use_High_Res_Component) and self.opt.SDF_Filter: 
            if self.opt.SDF_Filter_config==1 and (self.sdf_plane is not None) :
                error['Err(sdf_plane)'] = 0
                len_to_normalize_by = len(self.im_feat_list)  
                for i,im_feat in enumerate(self.im_feat_list):  # im_feat has shape of [B, C, H, W]

                    #if self.opt.SDF_Filter_subconfig_except_last and ( i == (len(self.im_feat_list)-1 ) ):
                    #    len_to_normalize_by = len_to_normalize_by-1
                    #    continue

                    if not self.opt.SDF_Filter_subconfig_fix_repair:
                        im_feat_plane = torch.sum(im_feat, dim = 1)
                        error['Err(sdf_plane)'] += self.criteria['sdf_plane'](im_feat_plane, self.sdf_plane)
                    else:
                        #channel_dim = im_feat.shape[1]
                        channel_dim = 120
                        im_feat_plane = torch.sum(im_feat, dim = 1)
                        error['Err(sdf_plane)'] += self.criteria['sdf_plane']( torch.div(im_feat_plane, channel_dim) , torch.div(self.sdf_plane, channel_dim) )
                
                if len_to_normalize_by != 0:
                    error['Err(sdf_plane)'] /= len_to_normalize_by       






        if self.opt.predict_vertex_normals and (not self.use_High_Res_Component) and self.opt.SDF_Filter:
            if self.opt.pred_vertex_normals_only_for_last_layer:
                dilution_factor = 1.0 / self.opt.num_stack_low_res
            else:
                dilution_factor = 1.0


            if self.opt.pred_vertex_normals_only_for_penultimate:
                error['Err(occ)']  = 0.45 * error['Err(occ)']  + 0.45 * error['Err(sdf_plane)'] + 0.10 * error['Err(normal_pred)']
            else:                
                error['Err(occ)']  = 0.5 * error['Err(occ)']  + 0.5 * error['Err(sdf_plane)'] + 0.1 * error['Err(normal_pred)'] * ( len(self.intermediate_preds_list_normal_pred) / self.opt.num_stack_low_res )


        elif self.opt.predict_vertex_normals and (not self.use_High_Res_Component):


            if self.opt.vertex_normals_config == 0:
                error['Err(occ)'] = 0.75 * error['Err(occ)'] + 0.25 * error['Err(normal_pred)']
            elif self.opt.vertex_normals_config == 1:
                error['Err(occ)'] = 0.5 * error['Err(occ)'] + 0.5 * error['Err(normal_pred)']
            elif self.opt.vertex_normals_config in [2,3] :
                error['Err(occ)'] = 0.65 * error['Err(occ)'] + 0.35 * error['Err(normal_pred)']
            else:
                raise Exception("Not yet implemented")


        elif (not self.use_High_Res_Component) and self.opt.SDF_Filter: 
            if self.opt.SDF_Filter_config == 1 and (self.sdf_plane is not None):
                if not self.opt.SDF_Filter_subconfig_fix_repair:
                    error['Err(occ)']  = 0.75 * error['Err(occ)']  + 0.25 * 0.05 * error['Err(sdf_plane)'] 
                else:
                    error['Err(occ)']  = 0.5 * error['Err(occ)']  + 0.5 * error['Err(sdf_plane)'] 



        else:
            pass 

        return error
        

    def forward(self, images, points, calibs, labels, labels_normal_pred=None, vertex_normals_sample_pts=None, points_nml=None, labels_nml=None, nmlF = None, nmlB = None, netG_output_map = None, mask_low_res_tensor=None, mask_high_res_tensor=None, sdf_plane = None ):
        self.filter(images, nmlF = nmlF, nmlB = nmlB, netG_output_map = netG_output_map, mask_low_res_tensor=mask_low_res_tensor, mask_high_res_tensor=mask_high_res_tensor, sdf_plane=sdf_plane)
        
        self.query(points, calibs, labels=labels, labels_normal_pred=labels_normal_pred, vertex_normals_sample_pts=vertex_normals_sample_pts )
        res = self.get_preds()
            
        err = self.get_error()

        return err, res





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





class modified_Tanh(nn.Module):
    def __init__(self, factor=1.5, translation=0.5):
        super().__init__()
        self.tanh = nn.Tanh()
        self.factor = factor
        self.translation = translation

    def forward(self, x):
        return self.tanh(x) * self.factor + self.translation







