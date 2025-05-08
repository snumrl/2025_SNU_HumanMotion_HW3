# This code is a simple and minimal BVH parser developed for the Human Motion course. It may not be compatible with other BVH files or skeleton structures.

import numpy as np
from scipy.spatial.transform import Rotation as R

# sim_skel = "walker2d"  or "humanoid3d_lowerbody" or "myoleg"
def load_bvh(filepath, sim_skel):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    joint_names = []
    joint_channel_start = {}  # maps joint name to starting index for its channels in the motion data
    joint_rot_order = {}      # maps joint name to its rotation order string (e.g. "XYZ")
    current_index = 0         # overall channel count
    header_complete = False
    motion_data = []
    frame_time = 0.0
    
    # Parse header and motion data.
    for line in lines:
        line = line.strip()
        # Once "MOTION" is encountered, switch to reading motion data.
        if line.startswith("MOTION"):
            header_complete = True
            continue
        if not header_complete:
            # Record joint names.
            if line.startswith("ROOT") or line.startswith("JOINT"):
                parts = line.split()
                joint_name = parts[1]
                joint_names.append(joint_name)
            elif line.startswith("CHANNELS"):
                parts = line.split()
                num_channels = int(parts[1])
                # For the ROOT joint, assume the first 3 channels are translation.
                if current_index == 0:
                    channel_tokens = parts[2:]  # e.g. ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"]
                    rot_tokens = channel_tokens[3:]  # skip translation channels
                    rot_order = "".join(tok[0] for tok in rot_tokens)
                    joint_channel_start[joint_names[0]] = current_index + 3
                    joint_rot_order[joint_names[0]] = rot_order
                    current_index += num_channels
                else:
                    # For other joints, all channels are rotation channels.
                    channel_tokens = parts[2:]
                    rot_order = "".join(tok[0] for tok in channel_tokens)
                    joint_channel_start[joint_names[-1]] = current_index
                    joint_rot_order[joint_names[-1]] = rot_order
                    current_index += num_channels
        else:
            # Skip lines that start with "Frame Time:" or "Frames:".
            if line.startswith("Frame Time:"):
                parts = line.split()
                frame_time = float(parts[2])
                continue
            if line.startswith("Frames:"):
                continue
            if line:
                frame = [float(x) for x in line.split()]
                motion_data.append(frame)
    
    if not motion_data:
        raise ValueError("No motion data found in BVH file.")
    
    motion_data = np.array(motion_data)
    joint_rotations = {}

    for joint in joint_names:
        start_idx = joint_channel_start[joint]
        order = joint_rot_order[joint]
        angles = motion_data[:, start_idx:start_idx+3]

        # Coordnate conversion
        mapping = {'X': 'Y', 'Y': 'Z', 'Z': 'X'}
        order = ''.join(mapping[c] for c in order)

        if sim_skel == "walker2d":
            joint_rotations[joint] = R.from_euler(order, angles, degrees=True).as_euler("XZY") 
        elif sim_skel == "humanoid3d_lowerbody":
            if joint == "Hips":   
                joint_rotations[joint] =  R.from_euler(order, angles, degrees=True).as_quat()
            elif joint == "Spine":
                joint_rotations[joint] =  R.from_euler(order, angles, degrees=True).as_euler("ZYX")
            else:
                joint_rotations[joint] =  R.from_euler(order, angles, degrees=True).as_euler("XZY")
        elif sim_skel == "myoleg":
            
            if joint == "Hips":   
                joint_rotations[joint] =  R.from_euler(order, angles, degrees=True).as_quat()
            elif joint == "Spine":
                joint_rotations[joint] =  R.from_euler(order, angles, degrees=True).as_euler("ZYX")
            else:
                order = joint_rot_order[joint]
                joint_rotations[joint] =  R.from_euler(order, angles, degrees=True).as_euler("XZY")
                
    joint_rotations["Trans"] = motion_data[:, :3] 
    
    if sim_skel == "walker2d":
        joint_rotations["Trans"][:, 1] -= 110.0
    elif sim_skel == "humanoid3d_lowerbody":
        joint_rotations["Trans"][:, 1] += 20    
    elif sim_skel == "myoleg":
        joint_rotations["Trans"][:, 1] += 0

    
    # Extract values for joints of the sim skeleton 
    if sim_skel == "humanoid3d_lowerbody":
        selected_idx = [("Trans", 2, 0.01), ("Trans", 0, 0.01), ("Trans", 1, 0.01),
                        ("Hips", 3, 1.0), ("Hips", 0, 1.0), ("Hips", 1, 1.0), ("Hips", 2, 1.0),
                        ("Spine", 0, 1.0), ("Spine", 1, 1.0), ("Spine", 2, 1.0),
                        ("RightUpLeg", 0, 1.0), ("RightUpLeg", 1, 1.0), ("RightUpLeg", 2, 1.0),
                        ("RightLeg", 2, -1.0), 
                        ("LeftUpLeg", 0, -1.0), ("LeftUpLeg", 1, -1.0), ("LeftUpLeg", 2, 1.0),
                        ("LeftLeg", 2, -1.0),
                    ]    
        selected_values = np.array([joint_rotations[joint][:, idx] * w for joint, idx, w in selected_idx])
        selected_values[0, :] -= selected_values[0, 0]
    
    elif sim_skel == "walker2d":
        selected_idx = [("Trans", 2, 0.01), ("Trans", 1, 0.01), ("Hips", 2, 1.0), 
                        ("RightUpLeg", 2, -1.0), ("RightLeg", 2, -1.0), ("RightFoot", 2, -1.0), 
                        ("LeftUpLeg", 2,-1.0), ("LeftLeg", 2,-1.00), ("LeftFoot", 2, -1.0)]
        selected_values = np.array([joint_rotations[joint][:, idx] * w for joint, idx, w in selected_idx])
        selected_values[0, :] -= selected_values[0, 0]
        selected_values[1, :] -= selected_values[1, 0]

    elif sim_skel == "myoleg":
        selected_values = np.zeros((35, len(motion_data)), dtype=np.float32)
        selected_idx_pair = [("Trans", 2, 0.009, 0), ("Trans", 0, 0.009, 1), ("Trans", 1, 0.009, 2), 
                            ("Hips", 3, 1.0, 3), ("Hips", 0, 1.0,4), ("Hips", 1, 1.0,5), ("Hips", 2, 1.0,6),
                            ("RightUpLeg", 0, -1.0, 7), 
                            ("RightUpLeg", 1, 1.0, 8),
                            ("RightUpLeg", 2, 1.0, 9),
                            ("LeftUpLeg", 0, -1.0, 21),
                            ("LeftUpLeg", 1, -1.0, 22),
                            ("LeftUpLeg", 2, -1.0, 23),
                            ("RightLeg", 0, 1.0, 12), 
                            ("LeftLeg", 0, 1.0, 26),
                            ("RightFoot", 0, -1.0, 15), 
                            ("LeftFoot", 0, -1.0, 29),
                            ]

        # put the values of selected idx pair which are the last idx of the selected_idx
        for joint, idx, w, selected_idx in selected_idx_pair:
            selected_values[selected_idx, :] = joint_rotations[joint][:, idx] * w
        
        # Rotate the root joint 90 degrees about the Z axis
        selected_values[0, :] -= selected_values[0, 0]
        selected_values[1, :] -= selected_values[1, 0]
        rot_90_z = R.from_euler('z', -90, degrees=True)
        rot_90_z_mat = rot_90_z.as_matrix()  
        selected_values[:3, :] = rot_90_z_mat @ selected_values[:3, :]
        quats_original = selected_values[3:7, :].T
        quats_xyzw = np.concatenate([quats_original[:, 1:], quats_original[:, [0]]], axis=1)
        rotated = R.from_quat(quats_xyzw)
        rotated_new = rot_90_z * rotated
        quats_new_xyzw = rotated_new.as_quat()  # (x, y, z, w)
        quats_new_wxyz = np.concatenate([quats_new_xyzw[:, [3]], quats_new_xyzw[:, :3]], axis=1).T  # shape: (4, N)
        selected_values[3:7, :] = quats_new_wxyz


    return selected_values, frame_time
    
class Motion ():
    def __init__(self, filepath, sim_skeleton):
        self.sim_skeleton = sim_skeleton
        self.filepath = filepath
        self.ref_poses, self.frame_time = load_bvh(self.filepath, self.sim_skeleton)
        self.ref_poses = self.ref_poses.astype(np.float32)
        
        self.frame_idx = 0
        self.num_frames = int(self.ref_poses.shape[1])

    def get_ref_poses(self, time):
        _frame_idx = int(time / self.frame_time)
        cycle_idx = _frame_idx // self.num_frames
        self.frame_idx = _frame_idx % self.num_frames
        ref_pos = self.ref_poses[:, self.frame_idx].copy()
        
        if self.sim_skeleton == "walker2d":
            ref_pos[0] = ref_pos[0] + (self.ref_poses[0, -1] - self.ref_poses[0, 1]) * cycle_idx
        elif self.sim_skeleton == "humanoid3d_lowerbody":
            ref_pos[0] = ref_pos[0] + (self.ref_poses[0, -1] - self.ref_poses[0, 1]) * cycle_idx
            ref_pos[1] = ref_pos[1] + (self.ref_poses[1, -1] - self.ref_poses[1, 1]) * cycle_idx
        elif self.sim_skeleton == "myoleg":
            ref_pos[0] = ref_pos[0] + (self.ref_poses[0, -1] - self.ref_poses[0, 1]) * cycle_idx
            ref_pos[1] = ref_pos[1] + (self.ref_poses[1, -1] - self.ref_poses[1, 1]) * cycle_idx
        
        return ref_pos
    
if __name__ == "__main__":
    filepath = "./asset/motions/walk.bvh"  # Update with your actual file path.
    poses = load_bvh(filepath)
    
    