import os
import json
import torch
import numpy as np
from typing import Dict, Tuple, Optional
from pytorch3d.transforms import matrix_to_quaternion
from scene_edit_utils import *

def compare_vehicle_trajectories(
    rigid_nodes,
    data_path: str,
    start_timestep: int = 0,
    end_timestep: Optional[int] = None
) -> Tuple[Dict, Dict]:
    """
    Compare vehicle trajectories between trained RigidNodes and NuPlan ground truth
    
    Args:
        rigid_nodes: Trained RigidNodes model
        data_path: Path to the processed NuPlan scene directory
        start_timestep: Starting frame index
        end_timestep: Ending frame index (exclusive)
        
    Returns:
        Tuple of (rigid_trajectories, nuplan_trajectories)
        Each dictionary contains:
        {
            instance_id: {
                'positions': array of shape (num_frames, 3),
                'rotations': array of shape (num_frames, 4),
                'frame_idx': list of frame indices where vehicle appears
            }
        }
    """
    # 1. Extract trajectories from RigidNodes
    rigid_trajectories = {}
    unique_ids = torch.unique(rigid_nodes.point_ids[..., 0])
    for instance_id in unique_ids:
        positions = rigid_nodes.instances_trans[:, instance_id]
        positions = positions[~(positions.sum(dim=-1) == 0)]
        rotations = rigid_nodes.instances_quats[:, instance_id]
        
        rigid_trajectories[instance_id.item()] = {
            'positions': positions.detach().cpu().numpy() if torch.is_tensor(positions) else positions,
            'rotations': rotations.detach().cpu().numpy() if torch.is_tensor(rotations) else rotations,
        }
    
    # 2. Load NuPlan ground truth trajectories
    instances_info_path = os.path.join(data_path, "instances", "instances_info.json")
    with open(instances_info_path, "r") as f:
        instances_info = json.load(f)
    
    # Get initial ego pose for alignment
    ego_to_world_start = np.loadtxt(
        os.path.join(data_path, "ego_pose", f"{start_timestep:03d}.txt")
    )
    
    nuplan_trajectories = {}
    for instance_id, info in instances_info.items():
        if info["class_name"] != "vehicle":
            continue
            
        frame_annos = info["frame_annotations"]
        frame_indices = frame_annos["frame_idx"]
        obj_to_worlds = frame_annos["obj_to_world"]
        
        positions = []
        rotations = []
        valid_frames = []
        
        # 只处理前300帧的数据
        if end_timestep is None:
            end_timestep = 300
            
        for frame_idx, obj_to_world in zip(frame_indices, obj_to_worlds):
            if frame_idx >= end_timestep:
                continue
            if frame_idx < start_timestep:
                continue
                
            obj_to_world = np.array(obj_to_world).reshape(4, 4)
            # Align to first ego pose
            obj_to_world = np.linalg.inv(ego_to_world_start) @ obj_to_world
            
            # Extract position and rotation
            position = obj_to_world[:3, 3]
            rot_matrix = torch.from_numpy(obj_to_world[:3, :3]).float()
            rotation = matrix_to_quaternion(rot_matrix).numpy()
            
            positions.append(position)
            rotations.append(rotation)
            valid_frames.append(frame_idx)
            
        if len(positions) > 0:
            nuplan_trajectories[instance_id] = {
                'positions': np.stack(positions),
                'rotations': np.stack(rotations),
                'frame_idx': valid_frames
            }
    
    return rigid_trajectories, nuplan_trajectories

def evaluate_trajectory_error(
    rigid_trajectories: Dict,
    nuplan_trajectories: Dict
) -> Dict:
    """
    Calculate error metrics between RigidNodes and ground truth trajectories
    
    Args:
        rigid_trajectories: Trajectories from RigidNodes
        nuplan_trajectories: Ground truth trajectories from NuPlan
        
    Returns:
        Dictionary containing error metrics
    """
    position_errors = []
    rotation_errors = []
    
    for nuplan_id, nuplan_data in nuplan_trajectories.items():
        # Find matching rigid trajectory with minimum distance
        min_dist = float('inf')
        matched_rigid_id = None
        
        for rigid_id, rigid_data in rigid_trajectories.items():
            # Compare positions at valid frames
            frame_indices = nuplan_data['frame_idx']
            rigid_pos = rigid_data['positions'][frame_indices]
            nuplan_pos = nuplan_data['positions']
            
            mean_dist = np.mean(np.linalg.norm(rigid_pos - nuplan_pos, axis=1))
            if mean_dist < min_dist:
                min_dist = mean_dist
                matched_rigid_id = rigid_id
        
        if matched_rigid_id is not None:
            # Calculate detailed errors for best match
            frame_indices = nuplan_data['frame_idx']
            rigid_pos = rigid_trajectories[matched_rigid_id]['positions'][frame_indices]
            rigid_rot = rigid_trajectories[matched_rigid_id]['rotations'][frame_indices]
            
            pos_error = np.linalg.norm(rigid_pos - nuplan_data['positions'], axis=1)
            # Simple angle difference for rotation error
            rot_error = np.arccos(np.clip(
                np.sum(rigid_rot * nuplan_data['rotations'], axis=1),
                -1.0, 1.0
            ))
            
            position_errors.append(pos_error)
            rotation_errors.append(rot_error)
    
    position_errors = np.concatenate(position_errors)
    rotation_errors = np.concatenate(rotation_errors)
    
    return {
        'mean_position_error': np.mean(position_errors),
        'median_position_error': np.median(position_errors),
        'mean_rotation_error': np.mean(rotation_errors),
        'median_rotation_error': np.median(rotation_errors)
    }

def visualize_trajectory_comparison(
    rigid_trajectories: Dict,
    nuplan_trajectories: Dict,
    output_path: Optional[str] = None
):
    """
    分别可视化 RigidNodes 和 NuPlan 的轨迹，生成两张独立的图
    """
    import matplotlib.pyplot as plt
    
    # Create output directory if needed
    if output_path:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot RigidNodes trajectories
    plt.figure(figsize=(12, 8))
    for rigid_id, data in rigid_trajectories.items():
        positions = data['positions'][:300]  # 只取前300帧
        plt.plot(
            positions[:, 0],
            positions[:, 1],
            '-',
            label=f'Vehicle {rigid_id}'
        )
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('RigidNodes Vehicle Trajectories')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(f'{output_path}_rigid.png')
    plt.close()
    
    # Plot NuPlan ground truth trajectories
    plt.figure(figsize=(12, 8))
    for nuplan_id, data in nuplan_trajectories.items():
        positions = data['positions']
        plt.plot(
            positions[:, 0],
            positions[:, 1],
            '-',
            label=f'Vehicle {nuplan_id}'
        )
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('NuPlan Ground Truth Vehicle Trajectories')
    plt.legend()
    plt.grid(True)
    
    if output_path:
        plt.savefig(f'{output_path}_nuplan.png')
    plt.close()


# 加载训练好的模型
example_idx = "2021.05.12.22.00.38_veh-35_01008_01518"
folder_path = "output/nuplan_example/" + example_idx
resume_from = folder_path + "/checkpoint_edit.pth"
trainer = load_trainer(resume_from)
rigid_nodes = trainer.models['RigidNodes']

# 对比轨迹
data_path = f"data/nuplan/processed/mini/{example_idx}"
rigid_trajectories, nuplan_trajectories = compare_vehicle_trajectories(
    rigid_nodes=rigid_nodes,
    data_path=data_path,
    start_timestep=0,
    end_timestep=300
)

# 计算误差
# errors = evaluate_trajectory_error(rigid_trajectories, nuplan_trajectories)
# print("Trajectory Errors:")
# for metric, value in errors.items():
#     print(f"{metric}: {value:.4f}")

# 可视化对比
visualize_trajectory_comparison(
    rigid_trajectories,
    nuplan_trajectories,
    output_path="./figure/comparison"
)