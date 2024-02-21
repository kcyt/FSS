
# Fine Structure-Aware Sampling (FSS)
Official Implementation of "Fine Structure-Aware Sampling: A New Sampling Training Scheme for Pixel-Aligned Implicit Models in Single-View Human Reconstruction" accepted in AAAI 2024 (Main Track).


Prerequisite:
Refer to "environment_setup_for_FSS" text file to install the conda environment, get the required rendered RGB images, and generate the predicted normal maps.

# To generate and store sample points using FSS. 
`conda activate fss`
`cd ./FSS/apps` \
`python generate_sample_pts_and_labels.py` \
The sample points will be saved in a folder under ./apps/results

# To train the main model:
`conda activate fss`
`cd ./FSS/apps` \
`python train_main.py` 
Model weights and results will be saved in a folder under ./apps/checkpoints and ./apps/results respectively.


