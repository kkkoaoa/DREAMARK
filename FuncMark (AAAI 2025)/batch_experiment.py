import os
import torch
import pyvista
import argparse
import torch.nn.functional as F
from glob import glob
from collections import defaultdict

from backend import *
from model import *
from util import *

data_root = "[root to shapenet_watertight]"


def wm_on_mesh(mesh):
    STEPS = V().cfg.linspace_steps
    vertices = torch.FloatTensor(mesh.vertices)
    vertices_sph = cartesian2spherical(vertices)
    normals = torch.FloatTensor(mesh.vertex_normals)
    thetas = torch.linspace(0, torch.pi, STEPS+1)
    phis = torch.linspace(-torch.pi, torch.pi, STEPS+1)
    
    for i in range(STEPS):
        for j in range(STEPS):
            msg_idx = (i * STEPS + j) % msg.shape[0]
            direction = msg[msg_idx].detach().cpu()
            theta_min, theta_max = thetas[i], thetas[i+1]
            phi_min, phi_max = phis[j], phis[j+1]
            mask = select_vertices(vertices_sph, theta_min, theta_max, phi_min, phi_max)

            v = vertices[mask]
            vn = normals[mask]
            v = v+global_D(v)*direction*vn
            vertices[mask] = v
    meshwm = trimesh.Trimesh(
        vertices=vertices.numpy(),
        faces=mesh.faces
    )
    return meshwm

def evaluation_metrics(sdf_wm):
    metrics = defaultdict(lambda:0)
    # xyz_wm, _, _ = get_surf_pcl(sdf_wm, 30000)
    # P2S = sdf(xyz_wm)[0].abs().mean()
    # metrics[f'P2S-INR'] = P2S
    # gt_msg_bit, msg_idx = sdf_wm.get_msg(xyz_wm)
    # gt_msg_bit = gt_msg_bit.view(xyz_wm.shape[0], 1).clamp(0, 1)
    # msg_bit = sdf(xyz_wm)[0] > 0
    # V().info(f"Acc: {(gt_msg_bit == msg_bit).sum()}/{xyz_wm.shape[0]}={(gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]}")
    # metrics[f'Acc-INR'] = (gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]
    # decoded_msg = torch.zeros_like(sdf_wm.msg)
    # decoded_cnt = torch.zeros_like(sdf_wm.msg)
    # for i in range(sdf_wm.msg.shape[0]):
    #     mask = (msg_idx == i)
    #     decoded_msg[i] += msg_bit[mask].sum()
    #     decoded_cnt[i] += mask.sum()
    # V().info("\t\t- Acc Detail:"+", ".join([f"{decoded_msg[i]}/{decoded_cnt[i]}" for i in range(sdf_wm.msg.shape[0])]))
    # decoded_msg = (decoded_msg / decoded_cnt).round().clamp(0, 1)
    # V().info(f"Dec Acc: {(decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]}")
    # metrics[f'DecAcc-INR'] = (decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]

    for res in [16, 32, 64]:
        xyz, _ = mise_plot(mesh_from_surf(sdf), res, 2)
        xyz = torch.FloatTensor(xyz).to(V().cfg.device)

        xyz_wm, _ = mise_plot(mesh_from_surf(sdf_wm), res, 2)
        xyz_wm =  torch.FloatTensor(xyz_wm).to(V().cfg.device)
        '''CD'''
        CD = chamfer_distance(xyz.unsqueeze(0), xyz_wm.unsqueeze(0))
        '''P2S'''
        P2S = sdf(xyz_wm)[0].abs().mean()
        '''Normal'''
        _, normal_wm = sdf_wm(xyz_wm)
        normal_wm = normal_wm / normal_wm.norm(dim=-1, keepdim=True)
        idx = knn_query(xyz.unsqueeze(0), xyz_wm.unsqueeze(0), 1)[0][0, :, 0]
        _, normal = sdf(xyz[idx])
        normal = normal / normal.norm(dim=-1, keepdim=True)
        Normal_Diff = (normal_wm - normal).norm(dim=-1).mean()
        V().info(f"CD: {CD}, P2S: {P2S}, Normal: {Normal_Diff}")
        metrics[f'CD-{res*4}'] = CD
        metrics[f'P2S-{res*4}'] = P2S
        metrics[f'Normal-{res*4}'] = Normal_Diff

        '''Acc'''
        gt_msg_bit, msg_idx = sdf_wm.get_msg(xyz_wm)
        gt_msg_bit = gt_msg_bit.view(xyz_wm.shape[0], 1).clamp(0, 1)
        msg_bit = sdf(xyz_wm)[0] > 0
        V().info(f"Acc: {(gt_msg_bit == msg_bit).sum()}/{xyz_wm.shape[0]}={(gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]}")
        metrics[f'Acc-{res*4}'] = (gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]
        decoded_msg = torch.zeros_like(sdf_wm.msg)
        decoded_cnt = torch.zeros_like(sdf_wm.msg)
        for i in range(sdf_wm.msg.shape[0]):
            mask = (msg_idx == i)
            decoded_msg[i] += msg_bit[mask].sum()
            decoded_cnt[i] += mask.sum()
        V().info("\t\t- Acc Detail:"+", ".join([f"{decoded_msg[i]}/{decoded_cnt[i]}" for i in range(sdf_wm.msg.shape[0])]))
        decoded_msg = (decoded_msg / decoded_cnt).round().clamp(0, 1)
        V().info(f"Dec Acc: {(decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]}")
        metrics[f'DecAcc-{res*4}'] = (decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]
    
    with open(os.path.join(V().cfg.output_dir, V().cfg.task_name, f'{sdf_wm.batch["name"]}.result-{V().cfg.msg_length}'), "w") as f:
        f.write('\n'.join([f"{k}={v}" for k, v in metrics.items()]))

def robustness(sdf_wm):
    metrics = defaultdict(lambda:0)
    gt_xyz_wm, _ = mise_plot(mesh_from_surf(sdf_wm), 32, 2)
    gt_xyz_wm = torch.FloatTensor(gt_xyz_wm).to(V().cfg.device)
    for kkk in range(7):
        if kkk==0: xyz_wm = gt_xyz_wm
        elif kkk==1: xyz_wm = gt_xyz_wm+torch.normal(mean=0, std=0.001, size=gt_xyz_wm.shape).to(V().cfg.device)
        elif kkk==2: xyz_wm = gt_xyz_wm+torch.normal(mean=0, std=0.01, size=gt_xyz_wm.shape).to(V().cfg.device)
        elif kkk==3: xyz_wm = learning_rotations(sdf_wm, alpha_g=True)
        elif kkk==4: xyz_wm = learning_rotations(sdf_wm, theta_g=True)
        elif kkk==5: xyz_wm = learning_rotations(sdf_wm, translation_g=True)
        elif kkk==6: xyz_wm = learning_rotations(sdf_wm, alpha_g=True, theta_g=True, translation_g=True)

        gt_msg_bit, msg_idx = sdf_wm.get_msg(xyz_wm)
        gt_msg_bit = gt_msg_bit.view(xyz_wm.shape[0], 1).clamp(0, 1)
        msg_bit = sdf(xyz_wm)[0] > 0
        V().info(f"Acc-{kkk}: {(gt_msg_bit == msg_bit).sum()}/{xyz_wm.shape[0]}={(gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]}")
        metrics[f"Acc-{kkk}"] = (gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]
        
        decoded_msg = torch.zeros_like(sdf_wm.msg)
        decoded_cnt = torch.zeros_like(sdf_wm.msg)
        for i in range(sdf_wm.msg.shape[0]):
            mask = (msg_idx == i)
            decoded_msg[i] += msg_bit[mask].sum()
            decoded_cnt[i] += mask.sum()
        V().info(f"\t\t- AccDetail-{kkk}:"+", ".join([f"{decoded_msg[i]}/{decoded_cnt[i]}" for i in range(sdf_wm.msg.shape[0])]))
        decoded_msg = (decoded_msg / decoded_cnt).round().clamp(0, 1)
        V().info(f"DecAcc-{kkk}: {(decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]}")
        metrics[f"DecAcc-{kkk}"] = (decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]

    with open(os.path.join(V().cfg.output_dir, V().cfg.task_name, f'{sdf_wm.batch["name"]}.rob-{V().cfg.msg_length}'), "w") as f:
        f.write('\n'.join([f"{k}={v}" for k, v in metrics.items()]))

def watermark_INR_from_mesh(mesh_path, sdf_wm=None):
    if sdf_wm is None:
        mesh = trimesh.load(mesh_path)
        mesh.vertices = pc_normalize(mesh.vertices)[0]

        mesh = wm_on_mesh(mesh)
        nsample = 500000
        on_surface_x, on_surface_face = mesh.sample(nsample, return_index=True)
        on_surface_normal = np.asarray(mesh.face_normals[on_surface_face], dtype=np.float32)
        on_surface_sdf = mesh_to_sdf(mesh, on_surface_x, 
            surface_point_method='scan',
            sign_method='normal',
            bounding_radius=None,
            scan_count=100,
            scan_resolution=400,
            normal_sample_count=11).reshape(-1, 1)

        off_surface_x = np.random.uniform(-1, 1, size=(nsample, 3))
        off_surface_normal = np.ones_like(off_surface_x) * -1
        off_surface_sdf = mesh_to_sdf(mesh, off_surface_x, 
            surface_point_method='scan',
            sign_method='normal',
            bounding_radius=None,
            scan_count=100,
            scan_resolution=400,
            normal_sample_count=11).reshape(-1, 1)

        if False:
            print((on_surface_sdf < 0).sum())
            mesh = pyvista.PolyData(on_surface_x)
            colors = np.zeros_like(on_surface_x)
            colors[on_surface_sdf < 0, 2] = 1
            colors[on_surface_sdf > 0, 0] = 5
            mesh["color"] = colors
            mesh["Normals"] = on_surface_normal
            p = pyvista.Plotter()
            p.add_mesh(mesh, scalars="color")
            p.add_mesh(mesh.glyph(geom=pyvista.Arrow(scale=0.01), orient="Normals"), color="black")
            p.show()
        
        sdfs = np.concatenate((on_surface_sdf, off_surface_sdf), axis=0).reshape(-1, 1)
        points = np.concatenate((on_surface_x, off_surface_x), axis=0).reshape(-1, 3)
        normals = np.concatenate((on_surface_normal, off_surface_normal), axis=0).reshape(-1, 3)

        batch = {
            "name": f'{os.path.basename(os.path.splitext(mesh_path)[0])}.wm_mesh',
            "points": torch.FloatTensor(points).to(V().cfg.device),
            "sdfs": torch.FloatTensor(sdfs).to(V().cfg.device),
            "normals": torch.FloatTensor(normals).to(V().cfg.device),
            "msg": msg,
            "linspace_steps": V().cfg.linspace_steps
        }
        sdf_wm = SDF_WM(msg=msg)
        sdf_wm.train_batch(batch)
    evaluation_metrics(sdf_wm)
    plot(sdf_wm, visual_normal=True, screenshot=os.path.join(V().cfg.output_dir, V().cfg.task_name, f"{sdf_wm.batch['name']}.png"))

def watermark_INR(name, sdf_wm=None):
    if sdf_wm is None:
        sdf_wm = SDF_WM(sdf=sdf, msg=msg)
        batch = sdf_wm.prepare_batch()
        sdf_wm.train_batch(batch)
    evaluation_metrics(sdf_wm)
    plot(sdf_wm, visual_normal=True, screenshot=os.path.join(V().cfg.output_dir, V().cfg.task_name, f"{sdf_wm.batch['name']}.png"))


def create_sdf(name):
    npy_path = f"dataset/data/{name}.npy"
    batch = np.load(npy_path, allow_pickle=True).item()
    batch['points'] = torch.asarray(batch['points']).float().view(-1, 3)
    batch['sdfs'] = torch.asarray(batch['sdfs']).float().view(-1, 1)
    batch['normals'] = torch.asarray(batch['normals']).float().view(-1, 3)
    sdf = SDF()
    sdf.train_batch(batch)
    plot(sdf, visual_normal=True, screenshot=os.path.join(V().cfg.output_dir, V().cfg.task_name, f"{batch['name']}.png"))
    return sdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_config(args)

    off_file_list = glob(f"{data_root}/*/2_watertight/*.off")
    metrics = defaultdict(Avg)
    for off_file in off_file_list:
        name = os.path.basename(os.path.splitext(off_file)[0])
        if os.path.exists(os.path.join(V().cfg.output_dir, V().cfg.task_name, f'{name}.wm.result-{V().cfg.msg_length}')):
            print(f"{off_file} finished")
            continue

        if not os.path.exists(f"dataset/data/{name}.npy"):
            print(f"{off_file} not ready")
            continue

        sdf_search_path = f'dataset/sdf/{name}.sdf'
        if os.path.exists(sdf_search_path):
            print(f"{off_file}.sdf exists")
            sdf = SDF(sdf_checkpoint=sdf_search_path)
            plot(sdf, visual_normal=True, screenshot=os.path.join(V().cfg.output_dir, V().cfg.task_name, f"{name}.png"))
        else:
            print(f'creating {name}.sdf')
            sdf = create_sdf(name)

        sdf_wm_search_path = f'dataset/wm-{V().cfg.msg_length}/{name}.wm.sdf'
        msg = torch.tensor(np.random.choice([1, -1], size=(V().cfg.msg_length, ))).to(V().cfg.device)
        if os.path.exists(sdf_wm_search_path):
        # if False:
            print(f"{off_file}.wm.sdf exists")
            sdf_wm = SDF_WM(wm_checkpoint=sdf_wm_search_path, sdf=sdf)
            # robustness(sdf_wm)
            watermark_INR(name, sdf_wm)
        else:
            print(f"creating {off_file}.wm.sdf")
            watermark_INR(name)

        sdf_wm_mesh_search_path = f'dataset/wmb-{V().cfg.msg_length}/{name}.wm_mesh.sdf'
        if os.path.exists(sdf_wm_mesh_search_path):
            print(f"{off_file}.wm_mesh.sdf exists")
            sdf_wm = SDF_WM(wm_checkpoint=sdf_wm_mesh_search_path, sdf=sdf)
            watermark_INR_from_mesh(off_file, sdf_wm)
        else:
            print(f"creating {off_file}.wm_mesh.sdf")
            watermark_INR_from_mesh(off_file)
    print('\n'.join(f'{k}:{v}' for k, v in metrics.items()))