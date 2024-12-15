from scene_edit_utils import *


def get_vehicle_positions(rigid):
    """
    Get overall position for each vehicle
    
    Args:
        rigid: RigidNodes object
        
    Returns:
        Dict containing vehicle positions across all frames
    """
    num_vehicles = rigid.instances_trans.shape[1]
    results = {}
    
    for veh_id in range(num_vehicles):
        results[veh_id] = {
            "positions": rigid.instances_trans[:, veh_id],
            "rotation": rigid.instances_quats[:, veh_id]
        }
    
    return results


example_idx = "2021.05.12.22.00.38_veh-35_01008_01518"

folder_path = "output/nuplan_example/" + example_idx

resume_from = folder_path + "/checkpoint_final.pth"
new_checkpoint_path = folder_path + "/checkpoint_edit.pth"

trainer_01518 = load_trainer(resume_from)
rigid_01518 = trainer_01518.models['RigidNodes']
vehicle_info = get_vehicle_positions(rigid_01518)
for veh_id, data in vehicle_info.items():
    print(f"Vehicle {veh_id}:")
    print(f"Position in first frame: {data['positions'][0]}")
    print(f"Rotation in first frame: {data['rotation'][0]}")


print(torch.unique(rigid_01518.point_ids[..., 0]))

