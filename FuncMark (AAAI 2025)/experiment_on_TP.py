import torch
import pyvista
import argparse
import torch.nn.functional as F
from glob import glob

from backend import *
from model import *
from util import *

data_root = "[root to shapenet_watertight]"
Nv = 100
Z_SCORE = 3.09
metrics = defaultdict(Avg)

nvs, z_scores = [], []

def do():
    '''ori'''
    xyz_wm, _, _ = get_surf_pcl(sdf, Nv)
    gt_msg_bit, msg_idx = sdf_wm.get_msg(xyz_wm)
    gt_msg_bit = gt_msg_bit.view(xyz_wm.shape[0], 1).clamp(0, 1)
    msg_bit = sdf(xyz_wm)[0] > 0
    z_score = ((gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0] - 0.5) * 2 * np.sqrt(Nv)
    print(z_score.item()<=Z_SCORE)
    metrics['TN'].update(z_score.item()<=Z_SCORE, 1)
    metrics['FP'].update(z_score.item()>Z_SCORE, 1)

    '''wm'''
    xyz_wm, _, _ = get_surf_pcl(sdf_wm, Nv)
    gt_msg_bit, msg_idx = sdf_wm.get_msg(xyz_wm)
    gt_msg_bit = gt_msg_bit.view(xyz_wm.shape[0], 1).clamp(0, 1)
    msg_bit = sdf(xyz_wm)[0] > 0
    acc = (gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]
    print(acc)
    z_score = ((gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0] - 0.5) * 2 * np.sqrt(xyz_wm.shape[0])
    nvs.append(xyz_wm.shape[0]); z_scores.append(z_score)
    metrics['FN'].update(z_score.item()<=Z_SCORE, 1)
    metrics['TP'].update(z_score.item()>Z_SCORE, 1)
    V().info('\n'.join([f'{k}:{v}' for k,v in metrics.items()]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_config(args)

    off_file_list = glob(f'{data_root}/*/2_watertight/*.off')
    for off_file in off_file_list:
        name = os.path.basename(os.path.splitext(off_file)[0])
        if os.path.exists(os.path.join(V().cfg.output_dir, V().cfg.task_name, f'{name}.wm_mesh.result-{V().cfg.msg_length}')):
            print(f"{off_file} finished")
            pass

        if not os.path.exists(f"dataset/data/{name}.npy"):
            print(f"{off_file} not ready")
            continue
        
        sdf_search_path = f'dataset/sdf/{name}.sdf'
        if os.path.exists(sdf_search_path):
            sdf = SDF(sdf_checkpoint=sdf_search_path)
        else:
            continue
        
        sdf_wm_searching_path = f'.output/batch_experiment-16-0.0005.yaml/{name}.wm.sdf'
        if os.path.exists(sdf_wm_searching_path):
            sdf_wm = SDF_WM(wm_checkpoint=sdf_wm_searching_path, sdf=sdf)
            do()
            with open(os.path.join(V().cfg.output_dir, V().cfg.task_name, f'{Nv}-{Z_SCORE}.result'), 'w') as f:
                f.write('\n'.join([f'{k}:{v}' for k,v in metrics.items()]))
        else:
            continue
    # with open('res_0.0005.txt', 'a') as f:
    #     for nv, z in zip(nvs, z_scores):
    #         f.write(f'{nv}={z}\n')