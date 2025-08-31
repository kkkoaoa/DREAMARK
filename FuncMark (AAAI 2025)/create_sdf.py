import os
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict

from model import *
from util import *
from common import *
from backend import *

def learning_rotations(surf):
    gt_y, _, _ = get_surf_pcl(surf, 1000)
    gt_theta = torch.randn(1, )
    gt_translation = torch.randn((1, 3)).cuda(V().cfg.device)
    gt_axis = torch.randn(3, )
    V().info(f"\n\t\t\t\t\t\t\tgt_theta={gt_theta}, \
                \n\t\t\t\t\t\t\tgt_translation={gt_translation.tolist()}")
    y, _ = rotation_A_t(gt_y, -gt_theta, gt_axis)
    y = y + gt_translation
    
    b = 3
    batch_size = b**6
    # y, _, _ = pc_normalize(y)
    theta = torch.linspace(0, 2*np.pi, b)
    xx, yy, zz = torch.linspace(-1, 1, b), torch.linspace(-1, 1, b), torch.linspace(-1, 1, b)
    p, t = torch.linspace(-torch.pi, torch.pi, b), torch.linspace(0, torch.pi, b)
    theta, xx, yy, zz, p, t = torch.meshgrid(theta, xx, yy, zz, p, t)
    r = torch.ones_like(p)
    theta = theta.reshape(-1, 1)
    translation = torch.vstack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1))).T
    axis = torch.vstack((r.reshape(-1), t.reshape(-1), p.reshape(-1))).T
    axis = spherical2cartesian(axis)
    # xx, yy, zz = torch.meshgrid(xx, yy, zz)
    # translation = torch.vstack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1))).T

    theta = torch.autograd.Variable(
        theta.view(batch_size, 1).to(V().cfg.device)
        # torch.linspace(0, 2*np.pi, batch_size).view(batch_size, 1) # [batch_size, 1]
    ).requires_grad_(True)
    translation = torch.autograd.Variable(
        translation.view(batch_size, 1, 3).to(V().cfg.device)
        # torch.randn(3, ).to(V().cfg.device)
        # (gt_translation+0.1).to(V().cfg.device)
    ).requires_grad_(True)
    axis = torch.autograd.Variable(
        axis.view(batch_size, 1, 3).to(V().cfg.device)
        # torch.FloatTensor([[1, p, t]])
        # torch.FloatTensor([1, 0.5, 0])
        # torch.FloatTensor(np.random.randn(3, ))
        # (gt_axis+0.1).to(V().cfg.device)
    ).requires_grad_(True)
    y = y.unsqueeze(0)

    opt = torch.optim.Adam([theta, translation, axis], lr=1e-2)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)
    for iteration in range(1000):
        opt.zero_grad()
        y_hat, R = rotation_A_t(y - translation, theta, axis)
        # y_hat = torch.matmul(y.unsqueeze(1), W).squeeze(1)
        # yR = torch.autograd.grad(y_hat, [W], grad_outputs=torch.ones_like(y_hat), retain_graph=True)[0]
        loss = surf(y_hat, with_grad=True)[0].abs().mean(dim=1)
        loss.backward(torch.ones_like(loss))
        opt.step()
        sch.step()
        V().info(f"Iteration {iteration}: \
                    \n\t\t\t\t\t\t\tloss.max={loss.max()}, \
                    \n\t\t\t\t\t\t\tloss.min={loss.min()}")
        # V().info(f'Iteration {iteration}: \
        #          \n\t\t\t\t\t\t\ttheta={theta}, \
        #          \n\t\t\t\t\t\t\ttranslation={translation.tolist()}, \
        #          \n\t\t\t\t\t\t\taxis={axis.tolist()}, \
        #          \n\t\t\t\t\t\t\tsdf.mean={self.sdf(y_hat).abs().mean().item()}, \
        #          \n\t\t\t\t\t\t\tsdf.max={self.sdf(y_hat).abs().max()}')

def visual_sampling():
    screenshot=None
    xyz, faces = mise_plot(mesh_from_surf(sdf), 64, 2)
    xyz, faces = xyz.astype(np.float32), faces.astype(np.int32)
    pts = get_surf_pcl(sdf, 10000)[0].cpu().numpy()
    
    p = pyvista.Plotter(off_screen=screenshot is not None)
    mesh = to_pyvista(xyz, faces)
    # pts = pyvista.PolyData(pts.cpu().numpy())
    p.add_mesh(mesh, color="silver")
    p.add_points(pts, render_points_as_spheres=True, color="black")
    cpos = None
    cpos = p.show(screenshot=screenshot, cpos=cpos)
    print(cpos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_config(args)
    

    if not args.debug:
        sdf = SDF()
        # batch = prepare_batch(V().cfg.model_path)
        batch = np.load(V().cfg.model_path, allow_pickle=True).item()
        batch['points'] = torch.asarray(batch['points']).float().view(-1, 3)
        batch['sdfs'] = torch.asarray(batch['sdfs']).float().view(-1, 1)
        batch['normals'] = torch.asarray(batch['normals']).float().view(-1, 3)
        sdf.train_batch(batch)
        plot(sdf, visual_normal=True)
    else:
        sdf = SDF(sdf_checkpoint=V().cfg.sdf_checkpoint)
        xyz = plot(sdf, 
            # visual_normal=True, 
            # screenshot="tmp.png"
        )
        # visual_sampling()
        # learning_rotations(surf)