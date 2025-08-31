import torch
import trimesh
import torch.nn.functional as F
from mesh_to_sdf import mesh_to_sdf, sample_sdf_near_surface

from funcwm.cu3d import sdf_by_normal

from common import *
from model.diff_opts import *

def sdf_loss(pred_sdf, gt):
    points = gt['points']
    normals = gt['normals']
    sdfs = gt['sdfs']
    
    gradient = torch.autograd.grad(pred_sdf, [points], grad_outputs=torch.ones_like(pred_sdf), create_graph=True)[0]

    sdf_constraint = torch.abs(pred_sdf - sdfs)
    # inter_constraint = torch.where(sdfs == 0, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(normals.norm(dim=-1, keepdim=True) == 1, 1 - F.cosine_similarity(gradient, normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    gradient_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    losses = {
        "sdf_constraint": torch.abs(sdf_constraint).mean() * 3e3,
        # "inter_constraint": inter_constraint.mean()        * 1e2,
        "normal_constraint": normal_constraint.mean()      * 1e2,
        "gradient_constraint": gradient_constraint.mean()  * 5e1,
    }
    # sdfs = gt['sdfs']
    # points = gt['points']
    # sdf_constraint = (pred_sdf - sdfs) ** 2
    # gradient = torch.autograd.grad(pred_sdf, [points], grad_outputs=torch.ones_like(pred_sdf), create_graph=True)[0]
    # gradient_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    # losses = {
    #     "sdf_constraint": sdf_constraint.mean() * 3e3,
    #     "gradient_constraint": gradient_constraint.mean() * 1e2,
    # }
    return losses

def cartesian2spherical(vertices, eps=1e-6):
    '''
    (x,y,z) --> (r, theta, phi)
        theta \in    [0, pi]
        phi   \in  [-pi, pi]
    V_ir = \sqrt(V_ix^2 + V_iy^2 + V_iz^2)
    V_itheta = arccos(V_iz / V_ir)
    V_iphi = arctan(V_iy / V_ix)
    '''
    Vir = torch.sqrt((vertices**2).sum(dim=-1))
    Vitheta = torch.arccos(torch.clamp(vertices[..., 2] / Vir, -1+eps, 1-eps))
    Viphi = torch.arctan2(vertices[..., 1], vertices[..., 0])
    return torch.stack([Vir, Vitheta, Viphi], dim=-1)

def spherical2cartesian(vertices):
    '''
    V_ix = V_ir * sin(V_itheta) * cos(V_iphi)
    V_iy = V_ir * sin(V_itheta) * sin(V_iphi)
    V_iz = V_ir * cos(V_itheta)
    '''
    Vix = vertices[..., 0] * torch.sin(vertices[..., 1]) * torch.cos(vertices[..., 2])
    Viy = vertices[..., 0] * torch.sin(vertices[..., 1]) * torch.sin(vertices[..., 2])
    Viz = vertices[..., 0] * torch.cos(vertices[..., 1])
    return torch.stack([Vix, Viy, Viz], dim=-1)

def select_vertices(vertices_sph, theta_min, theta_max, phi_min, phi_max):
    """
        [min, max]
    """
    mask_theta = torch.logical_and(vertices_sph[:,1] >= theta_min, vertices_sph[:,1] <= theta_max)
    mask_phi = torch.logical_and(vertices_sph[:,2] >= phi_min, vertices_sph[:,2] <= phi_max)
    return torch.logical_and(mask_theta, mask_phi)

def get_surf_pcl(surf, npoints, theta_min=None, theta_max=None, phi_min=None, phi_max=None, batch_size=100000, thr=1e-4, iteration=2):
    out = []
    cnt = 0
    while cnt < npoints:
        r = torch.rand(batch_size)
        if theta_min is not None:
            theta = torch.rand(batch_size)*(theta_max-theta_min)+theta_min
            phi = torch.rand(batch_size)*(phi_max-phi_min)+phi_min
            v_sph = torch.stack([r, theta, phi], dim=-1) # [batch_size, 3]
            v_cat = spherical2cartesian(v_sph).to(V().cfg.device)
        else:
            v_cat = torch.rand(batch_size, 3).to(V().cfg.device) * 2 - 1
        y = surf(v_cat)[0].abs().to(v_cat.device) # [batch_size, 1]
        mask = (y<thr).view(batch_size)
        mask_cnt = mask.sum()
        if mask_cnt < 1:
            continue
        v_eq = v_cat[mask]
        out.append(v_eq)
        cnt += mask_cnt
        print(f'get_surf_pcl: cnt={cnt}')
    rej_x = x = torch.cat(out, dim=0)[:npoints, :].detach()

    for i in range(iteration):
        F, g = surf(x)
        g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x = x - g * F

    after_bp = surf(x)[0].abs()
    before_bp = surf(rej_x)[0].abs()
    print(f'get_surf_pcl: after bp, {(after_bp<before_bp).sum()} out of {npoints} samples closer to surface than previous')
    x = torch.where((after_bp < before_bp).repeat(1, 3), x, rej_x)
    F, g = surf(x)
    g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    print(f'get_surf_pcl: sampled sdf mean {F.abs().mean()}')
    return x.detach(), g.detach(), F.detach()


def batched_T_reverse(func, yts, directions):
    """
    Suggest yts on cpu
    yts: shape (N, 3)
    directions: shape (N, 1)
    """
    yts = torch.as_tensor(yts).float()
    directions = torch.as_tensor(directions).view(yts.shape[0], 1)
    yt_split = torch.split(yts, 8192, dim=0)
    direction_split = torch.split(directions, 8192, dim=0)
    ys = []
    for yt, d in zip(yt_split, direction_split):
        yt, d = yt.to(V().cfg.device), d.to(V().cfg.device)
        y = T_reverse_with_retry(func, yt, d)
        ys.append(y.detach())
    ys = torch.cat(ys, dim=0)
    return ys.to(yts.device)

def T_reverse_with_retry(func, yt, direction):
    xts = [
        T_reverse(func, yt, direction),
        T_reverse(func, yt, direction),
        T_reverse(func, yt, direction),
    ]
    xt = torch.empty_like(xts[0])
    not_visited = torch.full((xt.shape[0], ), True).to(yt.device)
    for i, xti in enumerate(xts):
        xti_1 = xts[(i+1)%3]
        mask = torch.logical_and((xti_1 - xti).norm(dim=-1) < 1e-6, not_visited)
        xt[mask] = xti[mask]
        not_visited[mask] = False
    print(f'T_reverse_with_retry: not visied {not_visited.sum()}')
    xt[not_visited] = xts[0][not_visited]
    return xt



def T_reverse(func, yt, direction, batch_size=100, thr=1e-8, iteration=10):
    """
    Newton_raphson method to find solution x for yt = func(x)
    yt: shape (query_point, dim)
    direction: (query_point, 1)
    x: shape (query_point, batch_size, dim)
    y: shape (query_point, batch_size, dim)
    ret: shape (query_point, dim)
    """
    query_point, dim = yt.shape
    x = torch.rand(query_point, batch_size, dim).to(yt.device) * 2 - 1
    yt = yt.view(query_point, 1, dim)
    direction = direction.view(query_point, 1, 1)
    rest_eq = torch.arange(query_point).to(yt.device)
    ret = torch.empty((query_point, dim)).to(yt.device) # (query_point, dim)
    for i in range(iteration):
        x.requires_grad=True
        y = func(x, direction) - yt
        g, status = jacobian(y, x, create_graph=False) # g.shape=(query_point, batch_size, 3, 3) consumes too much GPU
        y = y.detach()
        assert status==0

        satisfied_mask = (y.norm(dim=-1) < thr).any(dim=-1)
        if satisfied_mask.sum():
            satisfied_eq = rest_eq[satisfied_mask]  # (?, )
            satisfied_y_set = y[satisfied_mask]     # (?, batch_size, dim)
            satisfied_x_set = x[satisfied_mask]     # (?, batch_size, dim)
            satisfied_x = satisfied_x_set[
                np.arange(satisfied_x_set.shape[0]), # (?, )
                torch.argmin(satisfied_y_set.norm(dim=-1), dim=1) # (?, )
            ]                                       # (?, dim)
            ret[satisfied_eq, :] = satisfied_x[:, :]
            
            unsatisfied_mask = torch.logical_not(satisfied_mask)
            rest_eq = rest_eq[unsatisfied_mask]
            x = x[unsatisfied_mask]
            y = y[unsatisfied_mask]
            direction = direction[unsatisfied_mask]
            g = g[unsatisfied_mask]
            yt = yt[unsatisfied_mask]
        
        if x.shape[0] == 0:
            break
        # print(f"Iteration {i} Finished: {query_point - x.shape[0]}/{query_point}, mean_y: {y.norm(dim=-1).min(dim=-1)[0].mean()}, max_y: {y.norm(dim=-1).min(dim=-1)[0].max()}")
        
        g_inv = torch.linalg.inv(g)
        x = (x - torch.matmul(
                g_inv, y.view(y.shape[0], batch_size, dim, 1)
                ).view(x.shape[0], batch_size, dim)
            ).detach()
    
    satisfied_x = x[
        np.arange(x.shape[0]),
        torch.argmin(y.norm(dim=-1), dim=1)
    ]                                       # (?, dim)
    ret[rest_eq, :] = satisfied_x[:, :]
    ret = ret.detach()
    del g, x, rest_eq, y
    torch.cuda.empty_cache()
    return ret


def learning_rotations(surf, alpha_g=False, theta_g=False, translation_g=False):
    gt_y, _ = mise_plot(mesh_from_surf(surf), 32, 1)
    gt_y =  torch.FloatTensor(gt_y).to(V().cfg.device)
    gt_theta = torch.randn(1, ) if theta_g else torch.FloatTensor([0])
    gt_translation = torch.randn((1, 3)).to(V().cfg.device) if translation_g else torch.FloatTensor([[0,0,0]]).to(V().cfg.device)
    gt_axis = torch.randn(3, )
    gt_alpha = torch.randn(1, 1).abs().to(V().cfg.device) if alpha_g else torch.FloatTensor([[1]]).to(V().cfg.device)
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
    V().info(f'learning_rotation: Optimal loss={optimal_loss}, optimal_idx={optimal_idx}, y-y_hat diff:{(y_hat-gt_y).norm(dim=-1).mean()}')

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
    
    loss = (y_hat_hat - gt_y).norm(dim=-1).mean()
    V().info(f"learning_rotation: loss={loss.item()}")
    return y_hat_hat.detach()