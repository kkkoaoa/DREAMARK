import os
import torch
import trimesh
import numpy as np
from mesh_to_sdf import mesh_to_sdf


from common import *
from model import *


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
            direction = msg[msg_idx]
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

def evaluation_metrics():
    xyz, _ = mise_plot(mesh_from_surf(sdf), 32, 2)
    xyz = torch.FloatTensor(xyz).to(V().cfg.device)

    xyz_wm, _ = mise_plot(mesh_from_surf(sdf_wm), 32, 2)
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

    '''Acc'''
    gt_msg_bit, msg_idx = sdf_wm.get_msg(xyz_wm)
    gt_msg_bit = gt_msg_bit.view(xyz_wm.shape[0], 1).clamp(0, 1)
    msg_bit = sdf(xyz_wm)[0] > 0
    V().info(f"Acc: {(gt_msg_bit == msg_bit).sum()}/{xyz_wm.shape[0]}={(gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]}")
    decoded_msg = torch.zeros_like(sdf_wm.msg)
    decoded_cnt = torch.zeros_like(sdf_wm.msg)
    for i in range(sdf_wm.msg.shape[0]):
        mask = (msg_idx == i)
        decoded_msg[i] += msg_bit[mask].sum()
        decoded_cnt[i] += mask.sum()
    V().info("\t\t- Acc Detail:"+", ".join([f"{decoded_msg[i]}/{decoded_cnt[i]}" for i in range(sdf_wm.msg.shape[0])]))
    decoded_msg = (decoded_msg / decoded_cnt).round().clamp(0, 1)
    V().info(f"Dec Acc: {(decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]}")

    with open(os.path.join(V().cfg.output_dir, V().cfg.task_name, f'{sdf_wm.batch["name"]}.result'), "w") as f:
        f.write(f"{CD.item()}\n\
            {P2S.item()}\n\
            {Normal_Diff.item()}\n\
            {(gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]}\n\
            {(decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_config(args)

    if not args.debug:
        msg = torch.tensor(np.random.choice([1, -1], size=(V().cfg.msg_length, )))
        mesh = trimesh.load(V().cfg.model_path)
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
            "name": f'{os.path.basename(os.path.splitext(V().cfg.model_path)[0])}.wm_mesh',
            "points": torch.FloatTensor(points).to(V().cfg.device),
            "sdfs": torch.FloatTensor(sdfs).to(V().cfg.device),
            "normals": torch.FloatTensor(normals).to(V().cfg.device),
            "msg": msg,
            "linspace_steps": V().cfg.linspace_steps
        }
        
        sdf = SDF_WM(msg=msg)
        sdf.train_batch(batch)
    else:
        sdf_wm = SDF_WM(sdf_checkpoint=V().cfg.sdf_checkpoint, wm_checkpoint=V().cfg.wm_checkpoint)
        sdf = sdf_wm.sdf
        # plot(sdf)
        evaluation_metrics()