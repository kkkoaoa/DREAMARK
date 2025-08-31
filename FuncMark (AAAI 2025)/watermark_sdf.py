import torch
import pyvista
import open3d as o3d
import argparse
from glob import glob
from matplotlib.colors import ListedColormap

from backend import *
from model import *
from util import *

# _tmp_cpos = [(0.08602293123973073, 0.11063668305405422, 0.0557371656660798),
#  (0.050902823471864214, 0.11834585719967075, 0.04351753910545217),
#  (0.1972796963166438, 0.9790372660051117, 0.05066313447122161)]

def ablation():
    deform = DeformNet().to(V().cfg.device)
    opt = torch.optim.Adam(deform.parameters(), lr=0.001)
    sch = torch.optim.lr_scheduler.StepLR(optimizer=opt, gamma=0.5, step_size=200)
    y, _, _ = get_surf_pcl(sdf, 1000)
    direction = sdf_t.get_msg(y)[0].view(y.shape[0], 1)
    x = sdf_t.T_forward(y, direction, create_graph=False)
    newton_y = batched_T_reverse(sdf_t.T_forward, x, direction)
    newton_invert_mse = (newton_y - y).norm(dim=-1).mean()
    best = torch.inf
    best_params = deform.state_dict()
    for iteration in range(1000):
        opt.zero_grad()
        pred_x = deform(y)
        loss = (pred_x - x).norm(dim=-1).mean()
        loss.backward()
        opt.step()
        sch.step()
        V().info(f"Iteration {iteration}: loss={loss.item()}")
        if loss < best:
            best = loss
            best_params = deform.state_dict()
    deform.load_state_dict(best_params)
    deform.eval()
    pred_x = deform(y)
    pred_y = deform.invert(x)
    forward_mse = (pred_x - x).norm(dim=-1).mean()
    invert_mse = (pred_y - y).norm(dim=-1).mean()
    print(f'newton_invert_mse: {newton_invert_mse}\nforward_mse: {forward_mse}\ninvert_mse: {invert_mse}')
    


def plot_acc(sdf_wm):
    xyz, faces = mise_plot(mesh_from_surf(sdf), 32, 2)
    xyz, faces = xyz.astype(np.float32), faces.astype(np.int32)
    moved_xyz, moved_faces = mise_plot(mesh_from_surf(sdf_wm), 32, 2)
    moved_xyz, moved_faces = moved_xyz.astype(np.float32), moved_faces.astype(np.int32)

    moved_xyz_tensor = torch.tensor(moved_xyz).cuda(V().cfg.device)

    p = pyvista.Plotter()
    mesh = to_pyvista(xyz, faces)
    p.add_mesh(mesh, color="black")

    bit_color = np.empty((2, 4))
    bit_color[0] = YELLOW_COLOR
    bit_color[1] = BLUE_COLOR
    bit_color = ListedColormap(bit_color)
    moved_mesh = to_pyvista(moved_xyz, moved_faces)
    gt_msg_bit = sdf_wm.get_msg(moved_xyz_tensor)[0]
    moved_mesh['color'] = (gt_msg_bit > 0).detach().cpu()
    p.add_mesh(
        moved_mesh,
        color="yellow",
        # scalars="color", cmap=bit_color, show_scalar_bar=False,
    )
    cpos = p.show(return_cpos=True)
    print(cpos)

def evaluation_metrics():
    xyz, _ = mise_plot(mesh_from_surf(sdf), 32, 2)
    xyz = torch.FloatTensor(xyz).to(V().cfg.device)

    xyz_wm, _ = mise_plot(mesh_from_surf(sdf_wm), 32, 2)
    xyz_wm =  torch.FloatTensor(xyz_wm).to(V().cfg.device)
    # gauss = torch.normal(mean=0, std=1e-2, size=xyz_wm.shape).to(V().cfg.device)
    # xyz_wm = xyz_wm + gauss

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


def learning_rotations(surf):
    gt_y, _ = mise_plot(mesh_from_surf(surf), 32, 1)
    gt_y =  torch.FloatTensor(gt_y).to(V().cfg.device)
    gt_theta = torch.randn(1, )
    gt_translation = torch.randn((1, 3)).to(V().cfg.device)
    gt_axis = torch.randn(3, )
    gt_alpha = torch.randn(1, 1).abs().to(V().cfg.device)
    V().info(f"\n\t\t\t\t\t\t\tgt_alpha={gt_alpha}, \
                \n\t\t\t\t\t\t\tgt_axis={gt_axis}, \
                \n\t\t\t\t\t\t\tgt_theta={gt_theta}, \
                \n\t\t\t\t\t\t\tgt_translation={gt_translation.tolist()}")
    y, _ = rotation_A_t(gt_y, -gt_theta, gt_axis)
    y = gt_alpha * y + gt_translation
    y = y + gt_translation
    
    b = 3
    batch_size = b**6
    y, _, _ = pc_normalize(y)
    theta = torch.linspace(0, 2*np.pi, b)
    xx, yy, zz = torch.linspace(-1, 1, b), torch.linspace(-1, 1, b), torch.linspace(-1, 1, b)
    p, t = torch.linspace(-torch.pi, torch.pi, b), torch.linspace(0, torch.pi, b)
    theta, xx, yy, zz, p, t = torch.meshgrid(theta, xx, yy, zz, p, t)
    r = torch.ones_like(p)
    theta = theta.reshape(-1, 1)
    translation = torch.vstack((xx.reshape(-1), yy.reshape(-1), zz.reshape(-1))).T
    axis = torch.vstack((r.reshape(-1), t.reshape(-1), p.reshape(-1))).T
    axis = spherical2cartesian(axis)

    theta = torch.autograd.Variable(
        theta.view(batch_size, 1).to(V().cfg.device)
    ).requires_grad_(True)
    translation = torch.autograd.Variable(
        translation.view(batch_size, 1, 3).to(V().cfg.device)
    ).requires_grad_(True)
    axis = torch.autograd.Variable(
        axis.view(batch_size, 1, 3).to(V().cfg.device)
    ).requires_grad_(True)
    y = y.unsqueeze(0)

    opt = torch.optim.Adam([theta, translation, axis], lr=1e-2)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5)
    for iteration in range(200):
        opt.zero_grad()
        y_hat, R = rotation_A_t(y, theta, axis)
        y_hat = y_hat + translation
        loss = surf(y_hat, with_grad=True)[0].abs().mean(dim=1)
        loss.backward(torch.ones_like(loss))
        opt.step()
        sch.step()
        V().info(f"Iteration {iteration}: \
                    \n\t\t\t\t\t\t\tloss.max={loss.max()}, \
                    \n\t\t\t\t\t\t\tloss.min={loss.min()}")
    tmp = torch.min(loss, dim=0)
    optimal_loss, optimal_idx = tmp.values, tmp.indices
    y_hat = y_hat[optimal_idx].view(-1, 3).detach()
    V().info(f'Optimal loss={optimal_loss}, optimal_idx={optimal_idx}, y-y_hat diff:{(y_hat-gt_y).norm(dim=-1).mean()}')

    theta = theta[optimal_idx].view(1, ).detach()
    translation = translation[optimal_idx].view(1, 3).detach()
    axis = axis[optimal_idx].view(3, ).detach()

    alpha = torch.autograd.Variable(
        torch.FloatTensor([[1]]).view(1, 1).to(V().cfg.device)
    ).requires_grad_(True)
    translation = torch.autograd.Variable(
        translation.view(1, 3).to(V().cfg.device)
    ).requires_grad_(True)
    opt = torch.optim.Adam([alpha, translation], lr=1e-2)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5)
    for iteration in range(400):
        opt.zero_grad()
        y_hat_hat = y_hat * alpha + translation
        loss = surf(y_hat_hat, with_grad=True)[0].abs().mean()
        loss.backward()
        opt.step()
        sch.step()
        V().info(f"Iteration {iteration}: \
                    \n\t\t\t\t\t\t\tloss.max={loss}")
    
    return y_hat


def robustness(sdf_wm):
    xyz_wm, _ = mise_plot(mesh_from_surf(sdf_wm), 32, 2)
    xyz_wm =  torch.FloatTensor(xyz_wm).to(V().cfg.device)
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
    
    '''test gauss'''
    xyz_wm = xyz_wm + torch.normal(0, 0.005, size=xyz_wm.shape).to(V().cfg.device)
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


    '''test rotations'''
    # xyz_wm = learning_rotations(sdf_wm)
    # gt_msg_bit, msg_idx = sdf_wm.get_msg(xyz_wm)
    # gt_msg_bit = gt_msg_bit.view(xyz_wm.shape[0], 1).clamp(0, 1)
    # msg_bit = sdf(xyz_wm)[0] > 0
    # V().info(f"Acc: {(gt_msg_bit == msg_bit).sum()}/{xyz_wm.shape[0]}={(gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]}")
    # decoded_msg = torch.zeros_like(sdf_wm.msg)
    # decoded_cnt = torch.zeros_like(sdf_wm.msg)
    # for i in range(sdf_wm.msg.shape[0]):
    #     mask = (msg_idx == i)
    #     decoded_msg[i] += msg_bit[mask].sum()
    #     decoded_cnt[i] += mask.sum()
    # V().info("\t\t- Acc Detail:"+", ".join([f"{decoded_msg[i]}/{decoded_cnt[i]}" for i in range(sdf_wm.msg.shape[0])]))
    # decoded_msg = (decoded_msg / decoded_cnt).round().clamp(0, 1)
    # V().info(f"Dec Acc: {(decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]}")

def _save_wm_mesh():
    xyz, faces = mise_plot(mesh_from_surf(sdf_wm), 32, 2)
    mesh = trimesh.Trimesh(xyz, faces)
    mesh.export("tmp.off")


def _decode_on_mesh():
    mesh = trimesh.load_mesh("dataset/remesh/Armadillo_128_32_remeshed.off")
    xyz = torch.FloatTensor(mesh.vertices).to(V().cfg.device)
    if False:
        visual_mesh(mesh.vertices, mesh.faces, screenshot="tmp.png")
    gt_msg_bit, msg_idx = sdf_wm.get_msg(xyz)
    gt_msg_bit = gt_msg_bit.view(xyz.shape[0], 1).clamp(0, 1)
    msg_bit = sdf(xyz)[0] > 0
    V().info(f"Acc: {(gt_msg_bit == msg_bit).sum()}/{xyz.shape[0]}={(gt_msg_bit == msg_bit).sum()/xyz.shape[0]}")
    decoded_msg = torch.zeros_like(sdf_wm.msg)
    decoded_cnt = torch.zeros_like(sdf_wm.msg)
    for i in range(sdf_wm.msg.shape[0]):
        mask = (msg_idx == i)
        decoded_msg[i] += msg_bit[mask].sum()
        decoded_cnt[i] += mask.sum()
    V().info("\t\t- Acc Detail:"+", ".join([f"{decoded_msg[i]}/{decoded_cnt[i]}" for i in range(sdf_wm.msg.shape[0])]))
    decoded_msg = (decoded_msg / decoded_cnt).round().clamp(0, 1)
    V().info(f"Dec Acc: {(decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]}")

def view_diff(sdf_wm):
    MAX_DISTORTION=0.01
    ploter_args = dict(off_screen=True)
    scalar_bar_args = dict(color='k', bold=True, title_font_size=20, label_font_size=32, font_family="arial", fmt="%.3f", title="",
            # background_color='#000000', fill=True,
        vertical=True, position_x=0.05, position_y=0.05, n_labels=6, height=0.85, width=0.10)
    res = 64
    xyz, face = mise_plot(mesh_from_surf(sdf), res, 2); xyz = torch.FloatTensor(xyz).to(V().cfg.device)
    xyz_wm, face_wm = mise_plot(mesh_from_surf(sdf_wm), res, 2); xyz_wm = torch.FloatTensor(xyz_wm).to(V().cfg.device)
    P2S = sdf(xyz_wm)[0].abs()
    print(P2S.mean())
    bit_color = np.empty((2, 4))
    bit_color[0] = YELLOW_COLOR
    bit_color[1] = BLUE_COLOR
    bit_color = ListedColormap(bit_color)

    mesh = to_pyvista(xyz.cpu().numpy(), face)
    mesh_wm = to_pyvista(xyz_wm.cpu().numpy(), face_wm)
    diff = to_pyvista(xyz_wm.cpu().numpy(), face_wm)
    diff['Distortions'] = P2S.cpu().numpy()
    diff['bit'] = (sdf(xyz_wm)[0] > 0).cpu().numpy()

    p = pyvista.Plotter()
    p.add_mesh(diff, scalars='Distortions', clim=[0, MAX_DISTORTION], scalar_bar_args=scalar_bar_args)
    cpos = p.show(return_cpos=True)
    print(cpos)
    return
    cpos = [(3.847619224887984, 2.268604564992705, 0.2509490210613253),
 (0.07579571429631668, 0.28169151498313333, -0.3634159475919552),
 (-0.44825296508663737, 0.6668712011591258, 0.59527479398646)]

    p = pyvista.Plotter(**ploter_args)
    p.add_mesh(diff, scalars='Distortions', clim=[0, MAX_DISTORTION], scalar_bar_args=scalar_bar_args)
    p.show(screenshot=os.path.join(V().cfg.output_dir, V().cfg.task_name, "xyz_diff.png"), cpos=cpos)

    # ori
    p = pyvista.Plotter(**ploter_args)
    p.add_mesh(mesh)
    p.show(screenshot=os.path.join(V().cfg.output_dir, V().cfg.task_name, "xyz.png"), cpos=cpos)
    
    # enc
    p = pyvista.Plotter(**ploter_args)
    p.add_mesh(mesh_wm)
    p.show(screenshot=os.path.join(V().cfg.output_dir, V().cfg.task_name, "xyz_enc.png"), cpos=cpos)

    # bit
    p = pyvista.Plotter(**ploter_args)
    p.add_mesh(diff, scalars='bit', show_scalar_bar=False, cmap=bit_color)
    p.show(screenshot=os.path.join(V().cfg.output_dir, V().cfg.task_name, "xyz_bit.png"), cpos=cpos)

def plot_local(sdf_wm):
    xyz, faces = mise_plot(mesh_from_surf(sdf), 32, 2)
    mesh = to_pyvista(xyz, faces)
    # watermarked
    mesh_wm = trimesh.load_mesh("dataset/remesh/322e8dccadea03d3340b9c9d10273ac_128_32.off"); mesh_wm = to_pyvista(mesh_wm.vertices, mesh_wm.faces)
    # gauss
    mesh_noise = mesh_wm.copy(); mesh_noise.points = mesh_wm.points+np.random.normal(0, 0.001, size=mesh_wm.points.shape)
    # remeshed
    mesh_wm_remeshed = trimesh.load_mesh("dataset/remesh/322e8dccadea03d3340b9c9d10273ac_128_32_remeshed.off"); mesh_wm_remeshed = to_pyvista(mesh_wm_remeshed.vertices, mesh_wm_remeshed.faces)
    # simplified
    mesh_smp = to_open3d(mesh_wm.points, mesh_wm.faces.reshape(-1, 4)[:,1:]).simplify_quadric_decimation(target_number_of_triangles=15000); mesh_smp = to_pyvista(np.asarray(mesh_smp.vertices), np.asarray(mesh_smp.triangles))
    # smooth
    mesh_smo = to_open3d(mesh_wm.points, mesh_wm.faces.reshape(-1, 4)[:,1:]).filter_smooth_laplacian(number_of_iterations=3); mesh_smo = to_pyvista(np.asarray(mesh_smo.vertices), np.asarray(mesh_smo.triangles))
    analyzed_mesh = mesh_wm_remeshed

    bit_color = np.empty((2, 4))
    bit_color[0] = ORANGE_COLOR
    bit_color[1] = BLUE_COLOR
    bit_color = ListedColormap(bit_color)

    xyz = torch.FloatTensor(xyz).to(V().cfg.device)
    xyz_wm = torch.FloatTensor(analyzed_mesh.points).to(V().cfg.device)
    P2S = sdf_wm(xyz_wm)[0].abs().mean()
    print(P2S)

    gt_msg_bit = (sdf_wm.get_msg(xyz_wm)[0] > 100000).detach().cpu()
    analyzed_mesh['color'] = gt_msg_bit
    p = pyvista.Plotter()
    p.add_mesh(mesh.slice(),
        show_edges=True, line_width=5,
        color='black')
    p.add_mesh(analyzed_mesh.slice(),
        show_edges=True, line_width=5,
        color="#FD9506",
        # scalars='color', cmap=bit_color, show_scalar_bar=False,
    )
    p.add_points(mesh.slice().points, render_points_as_spheres=True, point_size=15.0, color="black")
    p.add_points(analyzed_mesh.slice().points, render_points_as_spheres=True, point_size=15.0, 
                #  color="yellow",
                 color="#FD9506",
                #  scalars='color', cmap=bit_color, show_scalar_bar=False,
                 )

    cpos = p.show(cpos=None, return_cpos=True)
    print(cpos)

def decimation(sdf_wm):
    xyz, faces = mise_plot(mesh_from_surf(sdf_wm), 32, 2)
    mesh = to_open3d(xyz, faces)
    xyz = torch.FloatTensor(xyz).to(V().cfg.device)
    mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=1500)
    xyz_smp = torch.FloatTensor(mesh_smp.vertices).to(V().cfg.device)
    if False:
        visual_mesh(np.asarray(mesh_smp.vertices), np.asarray(mesh_smp.triangles), screenshot="tmp.png")
    
    xyz_wm = xyz_smp
    print(xyz_smp.shape[0]/xyz.shape[0])
    P2S = sdf_wm(xyz_wm)[0].abs().mean()
    print(P2S)

    gt_msg_bit, msg_idx = sdf_wm.get_msg(xyz_wm)
    gt_msg_bit = gt_msg_bit.view(xyz_wm.shape[0], 1).clamp(0, 1)
    msg_bit = sdf(xyz_wm)[0] > 0
    print(f"Acc: {(gt_msg_bit == msg_bit).sum()}/{xyz_wm.shape[0]}={(gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]}")
    decoded_msg = torch.zeros_like(sdf_wm.msg)
    decoded_cnt = torch.zeros_like(sdf_wm.msg)
    for i in range(sdf_wm.msg.shape[0]):
        mask = (msg_idx == i)
        decoded_msg[i] += msg_bit[mask].sum()
        decoded_cnt[i] += mask.sum()
    print("\t\t- Acc Detail:"+", ".join([f"{decoded_msg[i]}/{decoded_cnt[i]}" for i in range(sdf_wm.msg.shape[0])]))
    decoded_msg = (decoded_msg / decoded_cnt).round().clamp(0, 1)
    print(f"Dec Acc: {(decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]}")

def smoothing(sdf_wm):
    xyz, faces = mise_plot(mesh_from_surf(sdf_wm), 32, 2)
    mesh = to_open3d(xyz, faces)
    xyz = torch.FloatTensor(xyz).to(V().cfg.device)
    mesh_smo = mesh.filter_smooth_laplacian(number_of_iterations=3)
    xyz_smo = torch.FloatTensor(mesh_smo.vertices).to(V().cfg.device)
    if False:
        visual_mesh(np.asarray(mesh_smo.vertices), np.asarray(mesh_smo.triangles), screenshot="tmp.png")
    
    xyz_wm = xyz_smo
    P2S = sdf_wm(xyz_wm)[0].abs().mean()
    print(P2S)

    gt_msg_bit, msg_idx = sdf_wm.get_msg(xyz_wm)
    gt_msg_bit = gt_msg_bit.view(xyz_wm.shape[0], 1).clamp(0, 1)
    msg_bit = sdf(xyz_wm)[0] > 0
    print(f"Acc: {(gt_msg_bit == msg_bit).sum()}/{xyz_wm.shape[0]}={(gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]}")
    decoded_msg = torch.zeros_like(sdf_wm.msg)
    decoded_cnt = torch.zeros_like(sdf_wm.msg)
    for i in range(sdf_wm.msg.shape[0]):
        mask = (msg_idx == i)
        decoded_msg[i] += msg_bit[mask].sum()
        decoded_cnt[i] += mask.sum()
    print("\t\t- Acc Detail:"+", ".join([f"{decoded_msg[i]}/{decoded_cnt[i]}" for i in range(sdf_wm.msg.shape[0])]))
    decoded_msg = (decoded_msg / decoded_cnt).round().clamp(0, 1)
    print(f"Dec Acc: {(decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]}")

def quantization(sdf_wm):
    xyz, faces = mise_plot(mesh_from_surf(sdf_wm), 32, 2)
    bit = 8
    max_coord = xyz.max()
    xyz_wm = torch.FloatTensor(np.floor((xyz/max_coord)*(1<<bit))/(1<<bit)*max_coord).to(V().cfg.device)
    if True:
        visual_mesh(xyz_wm.cpu().detach().numpy(), faces, screenshot="tmp.png")


    gt_msg_bit, msg_idx = sdf_wm.get_msg(xyz_wm)
    gt_msg_bit = gt_msg_bit.view(xyz_wm.shape[0], 1).clamp(0, 1)
    msg_bit = sdf(xyz_wm)[0] > 0
    print(f"Acc: {(gt_msg_bit == msg_bit).sum()}/{xyz_wm.shape[0]}={(gt_msg_bit == msg_bit).sum()/xyz_wm.shape[0]}")
    decoded_msg = torch.zeros_like(sdf_wm.msg)
    decoded_cnt = torch.zeros_like(sdf_wm.msg)
    for i in range(sdf_wm.msg.shape[0]):
        mask = (msg_idx == i)
        decoded_msg[i] += msg_bit[mask].sum()
        decoded_cnt[i] += mask.sum()
    print("\t\t- Acc Detail:"+", ".join([f"{decoded_msg[i]}/{decoded_cnt[i]}" for i in range(sdf_wm.msg.shape[0])]))
    decoded_msg = (decoded_msg / decoded_cnt).round().clamp(0, 1)
    print(f"Dec Acc: {(decoded_msg==sdf_wm.msg.clamp(0,1)).sum()/sdf_wm.msg.shape[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    init_config(args)
    
    assert V().cfg.sdf_checkpoint

    if not args.debug:
        sdf_wm = SDF_WM(sdf_checkpoint=V().cfg.sdf_checkpoint)
        sdf_t = sdf_wm.sdf_t
        sdf = sdf_wm.sdf

        batch = sdf_wm.prepare_batch()
        sdf_wm.train_batch(batch)
        evaluation_metrics()
        plot(sdf_wm, visual_normal=True)
    else:
        sdf_wm = SDF_WM(sdf_checkpoint=V().cfg.sdf_checkpoint, wm_checkpoint=V().cfg.wm_checkpoint)
        sdf_t = sdf_wm.sdf_t
        sdf = sdf_wm.sdf
        # evaluation_metrics()
        # plot(sdf, visual_normal=False, 
        #     #  screenshot="tmp.png"
        #      )
        # plot_acc(sdf_wm)
        # learning_rotations(sdf_wm)
        # robustness(sdf_wm)
        # _save_wm_mesh()
        _decode_on_mesh()
        # view_diff(sdf_wm)
        # plot_local(sdf_wm)
        # decimation(sdf_wm)
        # smoothing(sdf_wm)
        # quantization(sdf_wm)
        # ablation()