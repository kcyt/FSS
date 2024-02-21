

import time 
import trimesh
import trimesh.proximity
import trimesh.sample
import numpy as np
import math
import os
os.environ["PYTHONUNBUFFERED"] = "1"

from PIL import Image


import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle

from numpy.linalg import inv


import logging
log = logging.getLogger('trimesh')
log.setLevel(40)




def get_chamfer_dist(src_mesh, tgt_mesh,  num_samples=10000):
    # Chamfer
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples) # src_surf_pts  has shape of (num_of_pts, 3)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt_mesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    #src_tgt_dist = src_tgt_dist.mean()
    #tgt_src_dist = tgt_src_dist.mean()
    src_tgt_dist = np.mean(np.square(src_tgt_dist))
    tgt_src_dist = np.mean(np.square(tgt_src_dist))

    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2

    return chamfer_dist



def get_surface_dist(src_mesh, tgt_mesh, num_samples=10000):
    # P2S
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0

    #src_tgt_dist = src_tgt_dist.mean()
    src_tgt_dist = np.mean(np.square(src_tgt_dist))

    return src_tgt_dist



def quick_get_chamfer_and_surface_dist(src_mesh, tgt_mesh,  num_samples=10000, sideview_confidence_box=None):
    # Chamfer
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples) # src_surf_pts  has shape of (num_of_pts, 3)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt_mesh, num_samples)

    if sideview_confidence_box is not None:
        sideview_confidence_box = torch.unsqueeze(sideview_confidence_box,0) 
        sideview_confidence_box = sideview_confidence_box.float()

        temp_src_surf_pts = torch.tensor(src_surf_pts)
        temp_tgt_surf_pts = torch.tensor(tgt_surf_pts)

        temp_src_surf_pts = torch.unsqueeze(temp_src_surf_pts,0) # [B==1, N, 3]
        temp_tgt_surf_pts = torch.unsqueeze(temp_tgt_surf_pts,0) # [B==1, N, 3]


        uv = temp_src_surf_pts   # [B==1, N, 3]
        #uv = uv.transpose(1, 2) 
        uv = uv[:, :, None, None, :] # [B, N, 1, 1, 3]
        uv = uv.float()

        sideview_confidence_box_values = torch.nn.functional.grid_sample(sideview_confidence_box[:, None, :, : ,:], uv, mode='bilinear' ) #[B==1, C==1, N, 1, 1]
        sideview_confidence_box_values = sideview_confidence_box_values[0,0,:,0,0] #(N,)
        sideview_confidence_box_values = sideview_confidence_box_values.detach().cpu().numpy()
    

            


    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts) # src_tgt_dist is of shape (N,)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts) # tgt_src_dist is of shape (N,)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    src_tgt_dist = np.square(src_tgt_dist)
    tgt_src_dist = np.square(tgt_src_dist)

    if sideview_confidence_box is not None:
        src_tgt_dist = src_tgt_dist * sideview_confidence_box_values
        tgt_src_dist = tgt_src_dist * sideview_confidence_box_values


    src_tgt_dist = np.mean(src_tgt_dist)
    tgt_src_dist = np.mean(tgt_src_dist)

    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2
    surface_dist = src_tgt_dist

    return chamfer_dist, surface_dist




"""
if __name__ == "__main__":

    run_test_mesh_already_prepared()
"""


