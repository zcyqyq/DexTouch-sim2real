import os, sys
os.chdir("/home/charliecheng/caiyi/DexGraspNet2")
sys.path.append(os.getcwd())
import numpy as np
import torch
from PIL import Image
import scipy.io as scio
from types import SimpleNamespace
import plotly.graph_objects as go
from src.utils.pc import depth_image_to_point_cloud, transform_pc
from src.utils.dataset import get_sparse_tensor
from src.utils.robot_info import GRIPPER_DEPTH_BASE, GRIPPER_NEW_DEPTH
from src.utils.util import set_seed
from src.utils.config import load_config
from src.network.model import get_model
import MinkowskiEngine as ME


def load_point_cloud(path, view, gt_str='_gt'):
    depth = np.array(Image.open(os.path.join(path, 'depth'+gt_str, str(view).zfill(4) + '.png')))
    meta = scio.loadmat(os.path.join(path, 'meta', str(view).zfill(4) + '.mat'))
    instrincs = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    cloud = depth_image_to_point_cloud(depth, instrincs, factor_depth)

    camera_poses = np.load(os.path.join(path, 'camera_poses.npy'))
    align_mat = np.load(os.path.join(path, 'cam0_wrt_table.npy'))
    trans = np.dot(align_mat, camera_poses[int(view)])
    cloud = cloud.reshape(-1, 3)
    cloud = transform_pc(cloud, trans)
    return cloud


def load_grasp_model(ckpt_path, batch_size=8):
    args = SimpleNamespace(
        batch_size=batch_size,
        ckpt=ckpt_path,
        type=None,
        split='test_novel',
        save=1,
        eval=1,
        proc=10
    )
    arg_mapping = [
        ('batch_size', ('batch_size', int, 32)),
        ('ckpt', ('ckpt', str, None)),
        ('type', ('model/type', str, None)),
        ('split', ('split', str, None)),
        ('save', ('save', int, 1)),
        ('eval', ('eval', int, 1)),
        ('proc', ('proc', int, 10)),
    ]
    set_seed(42)
    config = load_config(os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 'config.yaml'), arg_mapping, args)
    model = get_model(config.model)
    model.config.voxel_size = config.data.voxel_size
    ckpt = torch.load(config.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model, config


def sample_grasp(model, cloud, voxel_size):
    pc_tensor = torch.tensor(cloud, dtype=torch.float).unsqueeze(0)
    idxs = np.random.choice(len(cloud), 40000, replace=True)
    cloud_sampled = cloud[idxs]
    sparse_data = get_sparse_tensor(pc_tensor, voxel_size)
    sparse_data = {k: v.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for k, v in sparse_data.items()}
    
    with torch.no_grad():
        rot, trans, joint, log_prob, _ = model.sample(
            sparse_data, k=1024,
            edge=None,
            allow_fail=True,
            cate=False,
            ratio=0.1,
            graspness_scale=1,
            near=True
        )

    width = joint[..., 0]
    depth = torch.full_like(width, GRIPPER_NEW_DEPTH)
    grasp = torch.cat([
        log_prob[0][:, None], width[0][:, None],
        torch.full_like(width[0][:, None], GRIPPER_DEPTH_BASE),
        torch.full_like(depth[0][:, None], GRIPPER_NEW_DEPTH),
        rot[0].reshape(-1, 9), trans[0],
        torch.full_like(width[0][:, None], -1)
    ], dim=-1)
    return cloud_sampled, grasp


def get_best_grasp(grasp):
    best_idx = torch.argmax(grasp[:, 0]).item()
    return grasp[best_idx:best_idx+1]


def visualize_grasp_plotly_with_box(cloud, grasp_17, arrow_len=0.05):
    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    data = [go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=1.5, color='lightgray', opacity=0.4),
        name='Point Cloud'
    )]

    pos = np.array(grasp_17[13:16])
    data.append(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Grasp Center'
    ))

    rot = np.array(grasp_17[4:13]).reshape(3, 3)
    axis_colors = ['red', 'green', 'blue']
    for i in range(3):
        end = pos + rot[:, i] * arrow_len
        data.append(go.Scatter3d(
            x=[pos[0], end[0]], y=[pos[1], end[1]], z=[pos[2], end[2]],
            mode='lines',
            line=dict(width=5, color=axis_colors[i]),
            name=f'axis_{i}'
        ))

    width = grasp_17[1]
    depth = grasp_17[3]
    finger_size = np.array([0.01, width/2, depth])
    box_corners = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1]
    ]) * finger_size
    corners_world = box_corners @ rot.T + pos
    faces = np.array([
        [0,1,2], [0,2,3], [4,5,6], [4,6,7],
        [0,1,5], [0,5,4], [2,3,7], [2,7,6],
        [1,2,6], [1,6,5], [3,0,4], [3,4,7],
    ])
    data.append(go.Mesh3d(
        x=corners_world[:, 0],
        y=corners_world[:, 1],
        z=corners_world[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color='orange', opacity=0.3, name='Gripper Box'
    ))

    fig = go.Figure(data=data)
    fig.update_layout(scene=dict(aspectmode='data'), title='Grasp + Gripper Box',
                      margin=dict(l=0, r=0, t=30, b=0))
    fig.show()


# -------------------------
# Main pipeline
# -------------------------
if __name__ == "__main__":
    BASE_DIR = "/home/charliecheng/caiyi/DexGraspNet2"
    scene_path = "/home/charliecheng/caiyi/DexGraspNet2/data/scenes/scene_0100/realsense"
    view_id = 0
    cloud = load_point_cloud(scene_path, view=view_id)

    ckpt_path = os.path.join(BASE_DIR, "experiments/exp_gripper_ours/ckpt/DexGraspNet2.0-ckpts/OURS_gripper/ckpt/ckpt_50000.pth")
    model, config = load_grasp_model(ckpt_path)

    cloud_vis, grasp = sample_grasp(model, cloud, config.data.voxel_size)
    best_grasp = get_best_grasp(grasp)

    visualize_grasp_plotly_with_box(cloud_vis, best_grasp[0].detach().cpu().numpy())
