from typing import List, Optional
from omegaconf import OmegaConf
import os
import time
import json
import wandb
import logging
import argparse
import numpy as np

import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
    render_novel_views
)

def load_trainer(resume_from):
    logger = logging.getLogger()
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    log_dir = os.path.dirname(resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    #cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    #args.enable_wandb = False
    for folder in ["videos_eval", "metrics_eval"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device
    )

    # Resume from checkpoint
    trainer.resume_from_checkpoint(
        ckpt_path=resume_from,
        load_only_model=True
    )
    logger.info(
        f"Resuming training from {resume_from}, starting at step {trainer.step}"
    )

    return trainer

# Function to multiply two quaternions
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1[0, 0], q1[0, 1], q1[0, 2], q1[0, 3]
    w2, x2, y2, z2 = q2[0, 0], q2[0, 1], q2[0, 2], q2[0, 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)

# Function to rotate a specified vehicle's quats
def rotate_quat(rigid, veh_id, angles):

    quats = rigid.instances_quats.clone()
    return_quats = quats.clone()

    for i, quat in enumerate(quats[:, veh_id]):
        # Create a quaternion for gradual rotation by 'angle' around the Z-axis
        angle = angles[i]
        rotation_quat = torch.tensor([torch.cos(angle / 2), 0.0, 0.0, torch.sin(angle / 2)], device='cuda:0')
        
        # Apply the rotation to the current quaternion
        rotated_quat = quat_multiply(quat.view(1, 4), rotation_quat.view(1, 4))
        return_quats[i, veh_id] = rotated_quat

    rigid.instances_quats = torch.nn.Parameter(return_quats)

    return rigid

def change_trans(rigid, veh_id, new_trans):

    trans = rigid.instances_trans.clone()
    return_trans = trans.clone()

    for i, tran in enumerate(trans[:, veh_id]):
        return_trans[i, veh_id] = new_trans[i]
    
    rigid.instances_trans = torch.nn.Parameter(return_trans)

    return rigid

def change_trans_gradual(rigid, veh_id, start_index, end_index, end_point):

    trans = rigid.instances_trans.clone()
    return_trans = trans.clone()

    # Define the point where you want to start transitioning and the target point
    start_point = trans[start_index, veh_id]
    target_point = torch.tensor(end_point, device='cuda:0')

    # Number of remaining steps
    remaining_steps = end_index - start_index

    # Compute the incremental change per step
    delta = (target_point - start_point) / remaining_steps

    # Update the remaining values to gradually reach the target
    for i in range(start_index + 1, end_index + 1):
        return_trans[i, veh_id] = return_trans[i - 1, veh_id] + delta
    
    rigid.instances_trans = torch.nn.Parameter(return_trans)
        
    return rigid

def dupe_vehicle(rigid, veh_id):
    # Get unique vehicle IDs from point_ids
    unique_vehicle_ids, counts = torch.unique(rigid.point_ids[..., 0], return_counts=True)

    points = {}
    # Print the number of points per vehicle
    for vehicle_id, count in zip(unique_vehicle_ids, counts):
        points[int(vehicle_id.data)] = int(count.data)
        print(f"Vehicle ID: {vehicle_id}, Number of points: {count}")

    # Get the current number of instances
    num_instances = rigid.num_instances

    num_points_per_vehicle = points[veh_id]

    # Step 1: Duplicate the transformation and quaternion data of an existing vehicle
    new_trans = rigid.instances_trans.clone()
    new_quats = rigid.instances_quats.clone()

    # Step 2: Append the new vehicle's transformation and quaternion
    new_trans = torch.cat((new_trans, rigid.instances_trans[:, veh_id:veh_id + 1]), dim=1)  # Duplicate vehicle 1's transformation
    new_quats = torch.cat((new_quats, rigid.instances_quats[:, veh_id:veh_id + 1]), dim=1)  # Duplicate vehicle 1's quaternions

    existing_vehicle_quats = rigid._quats[rigid.point_ids[..., 0] == veh_id]
    new_quats_for_points = existing_vehicle_quats.repeat(num_points_per_vehicle // existing_vehicle_quats.shape[0], 1)
    new_quats_att = torch.cat((rigid._quats, new_quats_for_points), dim=0)

    # Step 3: Adjust the new vehicle's transformation (e.g., shift by 5 units on the x-axis)
    new_trans[:, num_instances] += torch.tensor([0, -9.0, 0.0], device='cuda:0')  # Modify this to avoid overlap

    # Step 4: Duplicate other necessary attributes like opacity, spherical harmonics, etc.
    # Opacity
    existing_vehicle_opacities = rigid._opacities[rigid.point_ids[..., 0] == veh_id]
    new_opacities_for_points = existing_vehicle_opacities.repeat(num_points_per_vehicle // existing_vehicle_opacities.shape[0], 1)
    new_opacities = torch.cat((rigid._opacities, new_opacities_for_points), dim=0)

    # Spherical Harmonics (Direct Colors and SH)
    new_features_dc = torch.cat((rigid._features_dc, rigid._features_dc[rigid.point_ids[..., 0] == veh_id]), dim=0)
    new_features_rest = torch.cat((rigid._features_rest, rigid._features_rest[rigid.point_ids[..., 0] == veh_id]), dim=0)

    # Point IDs: Assign a new unique ID to the new vehicle instance
    new_point_ids_for_vehicle = torch.full((num_points_per_vehicle, 1), num_instances, dtype=torch.long, device='cuda:0')
    new_point_ids = torch.cat((rigid.point_ids, new_point_ids_for_vehicle), dim=0)

    # Scales: Duplicate scales of the existing vehicle
    new_scales = torch.cat((rigid._scales, rigid._scales[rigid.point_ids[..., 0] == veh_id]), dim=0)

    # Means: Duplicate the means (Gaussian centers)
    new_means = torch.cat((rigid._means, rigid._means[rigid.point_ids[..., 0] == veh_id]), dim=0)

    # Frame Info (instances_fv): Duplicate the frame info for the new vehicle
    new_fv = torch.cat((rigid.instances_fv, rigid.instances_fv[:, veh_id:veh_id + 1]), dim=1)

    # Update the rigid attributes by modifying the underlying data of the parameters in place
    with torch.no_grad():  # We don't want to track gradients for these updates
        
        # Translations
        rigid.instances_trans.data = new_trans
        
        # Quaternions
        rigid.instances_quats.data = new_quats
        rigid._quats.data = new_quats_att
        
        # Opacities
        rigid._opacities.data = new_opacities
        
        # Direct colors (features_dc)
        rigid._features_dc.data = new_features_dc
        
        # Spherical harmonics (features_rest)
        rigid._features_rest.data = new_features_rest
        
        # Point IDs
        rigid.point_ids = new_point_ids  # Not a Parameter, so we can assign it directly
        
        # Scales
        rigid._scales.data = new_scales
        
        # Gaussian centers (means)
        rigid._means.data = new_means
        
        # Frame information (instances_fv)
        rigid.instances_fv = new_fv  # Not a Parameter, so we can assign it directly

    print(f"New number of instances: {rigid.num_instances}")

    return rigid

def transfer_veh(rigid_1, rigid_2, veh_id):
    # Get unique vehicle IDs from point_ids in rigid_2
    unique_vehicle_ids, counts = torch.unique(rigid_2.point_ids[..., 0], return_counts=True)

    points = {}
    # Print the number of points per vehicle in rigid_2
    for vehicle_id, count in zip(unique_vehicle_ids, counts):
        points[int(vehicle_id.data)] = int(count.data)
        print(f"Vehicle ID: {vehicle_id}, Number of points: {count}")

    num_points_per_vehicle = points[veh_id]

    # Get the current number of instances in rigid_1
    num_instances_rigid_1 = rigid_1.num_instances

    # Step 1: Copy transformation and quaternion data from rigid_2 to rigid_1
    new_trans = torch.cat((rigid_1.instances_trans.clone(), rigid_2.instances_trans[:, veh_id:veh_id + 1]), dim=1)
    new_quats = torch.cat((rigid_1.instances_quats.clone(), rigid_2.instances_quats[:, veh_id:veh_id + 1]), dim=1)

    existing_vehicle_quats = rigid_2._quats[rigid_2.point_ids[..., 0] == veh_id]
    new_quats_for_points = existing_vehicle_quats.repeat(num_points_per_vehicle // existing_vehicle_quats.shape[0], 1)
    new_quats_att = torch.cat((rigid_1._quats, new_quats_for_points), dim=0)

    # Step 2: Adjust transformations if necessary (you can apply modifications here if needed)

    # Step 3: Copy other attributes from rigid_2 to rigid_1
    # Opacity
    existing_vehicle_opacities = rigid_2._opacities[rigid_2.point_ids[..., 0] == veh_id]
    new_opacities_for_points = existing_vehicle_opacities.repeat(num_points_per_vehicle // existing_vehicle_opacities.shape[0], 1)
    new_opacities = torch.cat((rigid_1._opacities, new_opacities_for_points), dim=0)

    # Spherical Harmonics (Direct Colors and SH)
    new_features_dc = torch.cat((rigid_1._features_dc, rigid_2._features_dc[rigid_2.point_ids[..., 0] == veh_id]), dim=0)
    new_features_rest = torch.cat((rigid_1._features_rest, rigid_2._features_rest[rigid_2.point_ids[..., 0] == veh_id]), dim=0)

    # Point IDs: Assign a new unique ID to the vehicle being added to rigid_1
    new_point_ids_for_vehicle = torch.full((num_points_per_vehicle, 1), num_instances_rigid_1, dtype=torch.long, device='cuda:0')
    new_point_ids = torch.cat((rigid_1.point_ids, new_point_ids_for_vehicle), dim=0)

    # Scales
    new_scales = torch.cat((rigid_1._scales, rigid_2._scales[rigid_2.point_ids[..., 0] == veh_id]), dim=0)

    # Means (Gaussian centers)
    new_means = torch.cat((rigid_1._means, rigid_2._means[rigid_2.point_ids[..., 0] == veh_id]), dim=0)

    # Frame Info (instances_fv)
    new_fv = torch.cat((rigid_1.instances_fv, rigid_2.instances_fv[:, veh_id:veh_id + 1]), dim=1)

    # Step 4: Update the rigid_1 attributes by modifying the underlying data of the parameters
    with torch.no_grad():
        # Translations
        rigid_1.instances_trans.data = new_trans
        
        # Quaternions
        rigid_1.instances_quats.data = new_quats
        rigid_1._quats.data = new_quats_att
        
        # Opacities
        rigid_1._opacities.data = new_opacities
        
        # Direct colors (features_dc)
        rigid_1._features_dc.data = new_features_dc
        
        # Spherical harmonics (features_rest)
        rigid_1._features_rest.data = new_features_rest
        
        # Point IDs
        rigid_1.point_ids = new_point_ids  # Not a Parameter, so we can assign it directly
        
        # Scales
        rigid_1._scales.data = new_scales
        
        # Gaussian centers (means)
        rigid_1._means.data = new_means
        
        # Frame information (instances_fv)
        rigid_1.instances_fv = new_fv  # Not a Parameter, so we can assign it directly

    print(f"New number of instances in rigid_1: {rigid_1.num_instances}")

    return rigid_1

def save_checkpoint(trainer, new_checkpoint_path):
    torch.save(trainer.state_dict(only_model=True), new_checkpoint_path)
