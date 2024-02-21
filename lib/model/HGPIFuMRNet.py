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
import cv2

class HGPIFuMRNet(BasePIFuNet):
    '''
    HGPIFu uses stacked hourglass as an image encoder.
    '''

    def __init__(self, 
                 opt, 
                 netG = None,
                 projection_mode='orthogonal',
                 #criteria={'occ': CustomBCELoss()}
                 criteria={'occ': nn.MSELoss()}
                 #criteria={'occ': nn.BCELoss()},
                 ):
        super(HGPIFuMRNet, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria
            )

        self.opt = opt
        self.name = 'hg_pifu_high_res'

        in_ch = 3
        try:
            if opt.use_front_normal:
                in_ch += 3
            if opt.use_back_normal:
                in_ch += 3
        except:
            pass
            
        
        self.image_filter = HGFilter(opt.num_stack_high_res, opt.hg_depth_high_res, in_ch, opt.hg_dim_high_res, 
                                     opt.norm, 'no_down', False, no_first_down_sampling=self.opt.no_first_down_sampling)





        self.mlp = MLP(
            filter_channels=self.opt.mlp_dim_high_res,
            merge_layer=-1,
            res_layers=self.opt.mlp_res_layers_high_res,
            #norm=self.opt.mlp_norm,
            norm="no_norm",
            last_op=nn.Sigmoid())

        self.im_feat_list = []
        self.preds_interm = None
        self.preds_low = None
        self.w = None
        self.gamma = None

        self.intermediate_preds_list = []

        init_net(self)

        self.netG = netG

    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        for module in self.children():
            module.train(mode)
        if not self.opt.train_full_pifu:
            self.netG.eval()
        return self

    def filter_global(self, images, nmlF=None, nmlB = None, unrolled_smpl_features=None, mask_low_res_tensor=None):
        '''
        apply a fully convolutional network to images.
        the resulting feature will be stored.
        args:
            images: [B1, C, H, W]
        '''

        if self.opt.train_full_pifu or self.opt.update_low_res_pifu: # To train both PiFu or not. opt.train_full_pifu is False by default
            self.netG.filter(images, nmlF = nmlF, nmlB = nmlB, unrolled_smpl_features=unrolled_smpl_features, mask_low_res_tensor=mask_low_res_tensor)
        else:
            with torch.no_grad():
                self.netG.filter(images, nmlF = nmlF, nmlB = nmlB, unrolled_smpl_features=unrolled_smpl_features, mask_low_res_tensor=mask_low_res_tensor)

    def filter_local(self, images, rect=None, cropped_nmlF = None, cropped_nmlB = None, mask_high_res_tensor=None):
        '''
        apply a fully convolutional network to images(with the normal maps).
        the resulting feature will be stored.
        args:
            images: [B1, B2, C, H, W]  # shape of (batch_size, 1, 3, 1024, 1024) 
        '''


        self.mask_high_res_tensor = mask_high_res_tensor

        # getting the normal maps
        nmls = []
        try:
            if cropped_nmlF is not None and self.opt.use_front_normal:
                #nmls.append(self.netG.nmlF)
                nmls.append(cropped_nmlF)
            if cropped_nmlB is not None and self.opt.use_back_normal:
                #nmls.append(self.netG.nmlB)
                nmls.append(cropped_nmlB)
        except:
            pass

        if len(nmls): # opt.loadSizeBig is the size of the input image given to the High resolution Pifu, and the image size is 1024 by default
            # upsample the two normal maps
            if images.shape[-1]> nmls[0].shape[-1]:
                #nmls = nn.Upsample(size=(self.opt.loadSizeBig,self.opt.loadSizeBig), mode='bilinear', align_corners=True)(torch.cat(nmls,1))
                nmls = nn.Upsample(size=(images.shape[-1] , images.shape[-1] ), mode='bilinear', align_corners=True)(torch.cat(nmls,1))

            else:
                nmls=torch.cat(nmls, dim = 1)


            # it's kind of damn way.
            if rect is None:
                #images = torch.cat([images, nmls[:,None].expand(-1,images.size(1),-1,-1,-1)], 2)
                images = torch.cat([images, nmls], 1) 

                
            else:
                nml = []
                for i in range(rect.size(0)):
                    for j in range(rect.size(1)):
                        x1, y1, x2, y2 = rect[i,j]
                        tmp = nmls[i,:,y1:y2,x1:x2] # tmp is unused
                        nml.append(nmls[i,:,y1:y2,x1:x2])
                nml = torch.stack(nml, 0).view(*rect.shape[:2],*nml[0].size())
                images = torch.cat([images, nml], 2)




        if (len(images.shape) > 4 ): # images have shape of [batch, 1, channels, height, width]
            self.im_feat_list, self.normx = self.image_filter(images.view(-1,*images.size()[2:]))  # images.view(...) has shape of (batch_size * 1, 3, 512, 512) 
        else: # images have shape of [batch, channels, height, width]
            self.im_feat_list, self.normx = self.image_filter(images) # from the paper, each element in the self.im_feat_list shd have shape of [batch_size, 16,  512, 512] 
        if not self.training:
            self.im_feat_list = [self.im_feat_list[-1]]
        

    def new_query(self, points, calib_high_res, calib_low_res, labels=None):
        """
        given 3d points, we obtain 2d projection of these given the camera matrices.
        filter() needs to be called beforehand.
        the prediction is stored to self.preds

        points: [Batch, 3, Num_of_pts] 3d points in world space
        calibs: [Batch, 3, 4] or [Batch, 4, 4] calibration matrices for each image
        labels: [Batch, 1, N] ground truth labels (for supervision only)


        """


        xyz = self.projection(points, calib_high_res) # [Batch, 3, Num_of_pts]
        xy = xyz[:, :2, :] #[Batch, 2, Num_of_pts]

        if self.opt.use_mask_for_rendering_high_res and (self.mask_high_res_tensor is not None):
            mask_values = self.index(self.mask_high_res_tensor , xy) 

        # if the point is outside bounding box, return outside.
        in_bb = (xyz >= -1) & (xyz <= 1) # [Batch, 3, Num_of_pts]
        in_bb = in_bb[:, 0, :] & in_bb[:, 1, :]  # [Batch, Num_of_pts]
        in_bb = in_bb[:, None, :].detach().float() # [Batch, 1, Num_of_pts]

        is_zero_bool = (xyz == 0) # [B, 3, N]; remove the (0,0,0) point that has been used to discard unwanted sample pts
        is_zero_bool = is_zero_bool[:, 0, :] & is_zero_bool[:, 1, :] & is_zero_bool[:, 2, :] # [B, N]
        not_zero_bool = torch.logical_not(is_zero_bool)
        not_zero_bool = not_zero_bool[:, None, :].detach().float() # [B, 1, N]



        # predictions by the low-resolution pifu
        self.netG.query(points=points, calibs=calib_low_res)
        preds_low = torch.stack(self.netG.intermediate_preds_list,0) # each element in "intermediate_preds_list" has shape [B, 1, N].
        # preds_low has shape of [Num_of_list_elements_in_intermediate_preds_list, Batch_size, 1, Num_of_pts ]
        z_feat = self.netG.phi    # phi is activations from an intermediate layer of the MLP of the low-resolution PIFU. It has already gone through a round of indexing
        if ( (not self.opt.train_full_pifu) and (not self.opt.update_low_res_pifu) ) : # opt.train_full_pifu is False by default. It will decide whether to train the low-resolution PIFU as well.
            z_feat = z_feat.detach()


        if labels is not None:
            newlabels = in_bb * labels # has shape of [Batch, 1, Num_of_pts]
            newlabels = not_zero_bool * newlabels
            with torch.no_grad():
                num_of_pts = in_bb.size(2) 
                size_of_batch= in_bb.size(0)
                ws = num_of_pts / in_bb.view(size_of_batch,-1).sum(1) # the denominator "in_bb.view(size_of_batch,-1).sum(1)" gives the number of points that are inside the bb for each batch. The denominator has the shape of (batch_size,)
                # thus, ws has shape of (batch_size,) and it represents that the fraction or ratio of points that are inside the bounding box.

                size_of_batch = newlabels.size(0)
                also_size_of_batch = in_bb.size(0)
                # gammas = 1 - newlabels.view(size_of_batch,-1).sum(1) / in_bb.view(also_size_of_batch,-1).sum(1) # numerator (excluding the '1 - ') is the number of points that are inside the object mesh, and it has shape of (batch_size,). The denominator is the number of pts inside the bb, and it has shape of (batch_size,).
                # thus, gammas is 1 minus "the fraction of points in the bb that is inside the object mesh". Thus gamma is the fraction of points that is in the bb but not in the object mesh.

        
        


        intermediate_preds_list = []
        for j, im_feat in enumerate(self.im_feat_list): # Each element in self.im_feat_list has shape of [batch_size, 16, 512, 512] 
            
            point_local_feat_list = [self.index(im_feat, xy), z_feat] # z_feat has already gone through a round of indexing. 'point_local_feat_list' should have shape of [batch_size, 272, num_of_points] 
            point_local_feat = torch.cat(point_local_feat_list, 1) # No change. Shape of [batch_size, 272, num_of_points] 

            pred = self.mlp(point_local_feat)[0] # [Batch_size, 1, Num_of_pts]
            pred = in_bb * pred   # [Batch_size, 1, Num_of_pts]
            pred = not_zero_bool * pred
            if self.opt.use_mask_for_rendering_high_res and (self.mask_high_res_tensor is not None):
                pred = mask_values * pred

            intermediate_preds_list.append(pred)   

        preds_interm = torch.stack(intermediate_preds_list,0) # shape of [num_of_list_elements_in_intermediate_preds_list, Batch_size, 1, Num_of_pts ]
        preds = intermediate_preds_list[-1] # [Batch_size, 1, Num_of_pts]

        # predictions in the final layer for the High-resolution pifu
        self.preds = preds   # [Batch_size, 1, Num_of_pts]

        # all the predictions in all layers for the High-resolution pifu
        self.preds_interm = preds_interm   # [num_of_list_elements_in_intermediate_preds_list, Batch_size, 1, Num_of_pts]
        
        # all the predictions in all layers for the Low-resolution pifu
        self.preds_low = preds_low  # [Num_of_list_elements_in_intermediate_preds_list, Batch_size, 1, Num_of_pts]
        
        if labels is not None:
            self.w = ws  # (batch_size,)
            #self.gamma = gammas  # Shape of (batch_size,)
            self.labels = newlabels  # Shape of [Batch, 1, Num_of_pts]

    
    def query(self, points, calib_local, calib_global=None, transforms=None, labels=None):
        '''
        given 3d points, we obtain 2d projection of these given the camera matrices.
        filter needs to be called beforehand.
        the prediction is stored to self.preds
        args:
            points: [B1, B2, 3, N] 3d points in world space
            calibs_local: [B1, B2, 4, 4] calibration matrices for each image
            calibs_global: [B1, 4, 4] calibration matrices for each image
            transforms: [B1, 2, 3] image space coordinate transforms
            labels: [B1, B2, C, N] ground truth labels (for supervision only)
        return:
            [B, C, N] prediction
        '''


        if calib_global is not None:
            B = calib_local.size(1) # B refers to B2
        else:
            B = 1  # B refers to B2
            points = points[:,None]
            calib_global = calib_local # shape of [B1, 4, 4]
            calib_local = calib_local[:,None]  # [B1, 1, 4, 4]

        ws = []
        preds = []
        preds_interm = []
        preds_low = []
        gammas = []
        newlabels = []
        for i in range(B):
            xyz = self.projection(points[:,i], calib_local[:,i], transforms)
            
            xy = xyz[:, :2, :]

            # if the point is outside bounding box, return outside.
            in_bb = (xyz >= -1) & (xyz <= 1)
            in_bb = in_bb[:, 0, :] & in_bb[:, 1, :] 
            in_bb = in_bb[:, None, :].detach().float()

            # low resolution predictions
            self.netG.query(points=points[:,i], calibs=calib_global)
            # storing the actual predictions made by the low-resolution pifu in its final and intermediate layers:
            preds_low.append(torch.stack(self.netG.intermediate_preds_list,0)) 

            if labels is not None: 
                newlabels.append(in_bb * labels[:,i])
                with torch.no_grad():
                    ws.append(in_bb.size(2) / in_bb.view(in_bb.size(0),-1).sum(1))
                    gammas.append(1 - newlabels[-1].view(newlabels[-1].size(0),-1).sum(1) / in_bb.view(in_bb.size(0),-1).sum(1))

            z_feat = self.netG.phi    # phi is activations from an intermediate layer of the MLP of the low-resolution PIFU
            if ( (not self.opt.train_full_pifu) and (not self.opt.update_low_res_pifu) ) : # opt.train_full_pifu is False by default. It will decide whether to train the low-resolution PIFU as well.
                z_feat = z_feat.detach()
                        
            intermediate_preds_list = []
            for j, im_feat in enumerate(self.im_feat_list):
                point_local_feat_list = [self.index(im_feat.view(-1,B,*im_feat.size()[1:])[:,i], xy), z_feat]
                point_local_feat = torch.cat(point_local_feat_list, 1)
                pred = self.mlp(point_local_feat)[0]
                pred = in_bb * pred
                intermediate_preds_list.append(pred)

            preds_interm.append(torch.stack(intermediate_preds_list,0))
            preds.append(intermediate_preds_list[-1])

        self.preds = torch.cat(preds,0)
        self.preds_interm = torch.cat(preds_interm, 1) # first dim is for intermediate predictions
        self.preds_low = torch.cat(preds_low, 1) # first dim is for intermediate predictions
        
        if labels is not None:
            self.w = torch.cat(ws,0)
            self.gamma = torch.cat(gammas,0)
            self.labels = torch.cat(newlabels,0)

    


    def calc_normal(self, points, calib_local, calib_global, transforms=None, labels=None, delta=0.001, fd_type='forward'):
        '''
        return surface normal in 'model' space.
        it computes normal only in the last stack.
        note that the current implementation use forward difference.
        args:
            points: [B1, B2, 3, N] 3d points in world space
            calibs_local: [B1, B2, 4, 4] calibration matrices for each image
            calibs_global: [B1, 4, 4] calibration matrices for each image
            transforms: [B1, 2, 3] image space coordinate transforms
            labels: [B1, B2, 3, N] ground truth normal
            delta: perturbation for finite difference
            fd_type: finite difference type (forward/backward/central) 
        '''
        B = calib_local.size(1)

        if labels is not None:
            self.labels_nml = labels.view(-1,*labels.size()[2:])

        im_feat = self.im_feat_list[-1].view(-1,B,*self.im_feat_list[-1].size()[1:])

        nmls = []
        for i in range(B):
            points_sub = points[:,i]
            pdx = points_sub.clone()
            pdx[:,0,:] += delta
            pdy = points_sub.clone()
            pdy[:,1,:] += delta
            pdz = points_sub.clone()
            pdz[:,2,:] += delta

            points_all = torch.stack([points_sub, pdx, pdy, pdz], 3)
            points_all = points_all.view(*points_sub.size()[:2],-1)
            xyz = self.projection(points_all, calib_local[:,i], transforms)
            xy = xyz[:, :2, :]


            self.netG.query(points=points_all, calibs=calib_global, update_pred=False)
            z_feat = self.netG.phi
            if not self.opt.train_full_pifu:
                z_feat = z_feat.detach()

            point_local_feat_list = [self.index(im_feat[:,i], xy), z_feat]            
            point_local_feat = torch.cat(point_local_feat_list, 1)
            pred = self.mlp(point_local_feat)[0]

            pred = pred.view(*pred.size()[:2],-1,4) # (B, 1, N, 4)

            # divide by delta is omitted since it's normalized anyway
            dfdx = pred[:,:,:,1] - pred[:,:,:,0]
            dfdy = pred[:,:,:,2] - pred[:,:,:,0]
            dfdz = pred[:,:,:,3] - pred[:,:,:,0]

            nml = -torch.cat([dfdx,dfdy,dfdz], 1)
            nml = F.normalize(nml, dim=1, eps=1e-8)

            nmls.append(nml)
        
        self.nmls = torch.stack(nmls,1).view(-1,3,points.size(3))

    def get_im_feat(self):
        '''
        return the image filter in the last stack
        return:
            [B, C, H, W]
        '''
        return self.im_feat_list[-1]

    def get_error(self):
        '''
        return the loss given the ground truth labels and prediction
        '''

        error = {}
        if self.opt.train_full_pifu:
            if not self.opt.no_intermediate_loss:
                error['Err(occ)'] = 0.0
                for i in range(self.preds_low.size(0)):
                    #error['Err(occ)'] += self.criteria['occ'](self.preds_low[i], self.labels, self.gamma, self.w)
                    error['Err(occ)'] += self.criteria['occ'](self.preds_low[i], self.labels)

                error['Err(occ)'] /= self.preds_low.size(0)

            error['Err(occ:fine)'] = 0.0
            for i in range(self.preds_interm.size(0)):
                #error['Err(occ:fine)'] += self.criteria['occ'](self.preds_interm[i], self.labels, self.gamma, self.w)
                error['Err(occ:fine)'] += self.criteria['occ'](self.preds_interm[i], self.labels)

            error['Err(occ:fine)'] /= self.preds_interm.size(0)

            # errors for the normal maps
            if self.nmls is not None and self.labels_nml is not None:
                error['Err(nml:fine)'] = self.criteria['nml'](self.nmls, self.labels_nml)
        else:
            error['Err(occ:fine)'] = 0.0
            for i in range(self.preds_interm.size(0)):
                #error['Err(occ:fine)'] += self.criteria['occ'](self.preds_interm[i], self.labels, self.gamma, self.w)
                error['Err(occ:fine)'] += self.criteria['occ'](self.preds_interm[i], self.labels)

            error['Err(occ:fine)'] /= self.preds_interm.size(0)


        return error


    def forward(self, images_local, images_global, points, calib_local, calib_global, labels, points_nml=None, labels_nml=None, rect=None, nmlF = None, nmlB = None, cropped_nmlF=None, cropped_nmlB= None, unrolled_smpl_features=None):

        self.filter_global(images_global, nmlF = nmlF, nmlB = nmlB, unrolled_smpl_features=unrolled_smpl_features) # execute forward pass of the low-resolution pifu
        
        # note that rect is always 'none' below.
        self.filter_local(images_local, rect, cropped_nmlF = cropped_nmlF, cropped_nmlB = cropped_nmlB) # execute forward pass of the high-resolution pifu 
        #calib_local is equal to calib_global, so the self.new_query only needs one of them.

        self.new_query(points, calib_high_res=calib_local,calib_low_res=calib_global, labels=labels)
        

        res = self.get_preds()
            
        err = self.get_error()

        return err, res
