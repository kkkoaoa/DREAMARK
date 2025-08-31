
from typing import Any
from tqdm import tqdm
from collections import defaultdict
from model.wm_util import *

from common import *

def global_D(y):
    return 0.001

def get_interval(y):
    STEPS = V().cfg.linspace_steps
    thetas = torch.linspace(0, torch.pi, STEPS + 1).cuda(V().cfg.device)
    phis = torch.linspace(-torch.pi, torch.pi, STEPS + 1).cuda(V().cfg.device)
    y_sph = cartesian2spherical(y)
    y_theta_i = torch.searchsorted(thetas, y_sph[..., 1], right=True).clamp(0, STEPS) - 1
    y_phi_i = torch.searchsorted(phis, y_sph[..., 2], right=True).clamp(0, STEPS) - 1
    assert (y_theta_i<0).sum()==0 and (y_phi_i<0).sum()==0 \
        and (y_theta_i>=STEPS).sum()==0 and (y_phi_i>=STEPS).sum()==0
    return thetas[y_theta_i], thetas[y_theta_i+1], phis[y_phi_i], phis[y_phi_i+1], y_sph

def global_D_2(y):
    theta_min, theta_max, phi_min, phi_max, y_sph = get_interval(y)
    mask = torch.logical_or(y[..., 0]!=0, y[..., 1]!=0)
    res = torch.zeros(y.shape[:-1], device=y.device)
    theta_int = theta_max - theta_min
    phi_int = phi_max - phi_min
    y_theta = y_sph[..., 1]
    y_phi = y_sph[..., 2]
    strength = torch.min((y_theta - theta_min)/theta_int, (theta_max - y_theta)/theta_int)
    strength = torch.min(strength, (phi_max - y_phi)/phi_int)
    strength = torch.min(strength, (y_phi - phi_min)/phi_int)
    res[mask] = strength[mask]
    return 0.001 * res.unsqueeze(-1)

class SDF:
    def __init__(self, sdf=None, sdf_checkpoint=None):
        self.sdf = sdf
        if sdf is None:
            self.sdf = V().cfg.sdf().cuda(V().cfg.device)
        self.opt = torch.optim.Adam(
            self.sdf.parameters(),
            lr=V().cfg.lr)

        self.scheduler = V().cfg.scheduler(self.opt)

        self.points_batch_size = V().cfg.points_batch_size
        self.load_sdf(sdf_checkpoint)

    def load_sdf(self, path):
        if path:
            self.batch = checkpoint = torch.load(path, map_location=f'cuda:{V().cfg.device}')
            self.sdf.load_state_dict(checkpoint['sdf'])

    def save_sdf(self, batch):
        checkpoint = {
            "sdf": self.sdf.state_dict(),
            **batch
        }
        torch.save(checkpoint, os.path.join(V().cfg.output_dir, V().cfg.task_name, f"{batch['name']}.sdf"))

    def get_msg(self, y):
        msg = self.batch['msg']
        y_sph = cartesian2spherical(y)
        y_theta_i = torch.searchsorted(self.thetas, y_sph[..., 1], right=True).clamp(0, self.linspace_steps) - 1
        y_phi_i = torch.searchsorted(self.phis, y_sph[..., 2], right=True).clamp(0, self.linspace_steps) - 1
        msg_idx = (y_theta_i * self.linspace_steps + y_phi_i) % msg.shape[0]
        return msg[msg_idx].detach(), msg_idx

    def __call__(self, x, with_grad=False, create_graph=False, with_normal=True):
        if create_graph: with_grad=True
        if with_grad:
            if x.is_leaf:
                x.requires_grad = True
            F = self.sdf(x)
            if with_normal:
                Fx = torch.autograd.grad(F, [x], grad_outputs=torch.ones_like(F), retain_graph=True, create_graph=create_graph)[0]
                return F, Fx
            else:
                return F, None
        else:
            x_split = torch.split(x, self.points_batch_size, dim=0)
            F, Fx = [], []
            for xi in x_split:
                if xi.is_leaf:
                    xi.requires_grad = True
                Fi = self.sdf(xi)
                Fxi = torch.autograd.grad(Fi, [xi], grad_outputs=torch.ones_like(Fi), create_graph=False)[0]
                F.append(Fi.detach())
                Fx.append(Fxi.detach())
            F = torch.concat(F, dim=0)
            Fx = torch.concat(Fx, dim=0)
            return F.detach(), Fx.detach()

    def train_batch(self, batch):
        best = np.inf
        self.batch = batch

        progress = tqdm(range(V().cfg.iteration))
        for iteration in progress:
            V().info(f"Epoch {iteration} Start: lr={self.opt.param_groups[0]['lr']}")

            self.opt.zero_grad()
            metrics = defaultdict(Avg)
            x_split = torch.split(batch["points"], self.points_batch_size, dim=0)
            sdf_split = torch.split(batch['sdfs'], self.points_batch_size, dim=0)
            normal_split = torch.split(batch["normals"], self.points_batch_size, dim=0)
            for (x, sdf, normal) in zip(x_split, sdf_split, normal_split):
                x = x.cuda(V().cfg.device).requires_grad_(True)
                sdf = sdf.cuda(V().cfg.device)
                normal = normal.cuda(V().cfg.device)

                pred_sdf = self.sdf(x)
                losses = sdf_loss(pred_sdf, {
                    "points": x,
                    "sdfs": sdf,
                    "normals": normal
                })
                tot_loss = sum(losses.values())
                tot_loss.backward()
                for k, v in losses.items():
                    metrics[k].update(v.item() * x.shape[0], x.shape[0])
                metrics['tot_loss'].update(tot_loss.item() * x.shape[0], x.shape[0])
            self.opt.step()

            V().info("metrics", **metrics)
            progress.set_postfix_str(
                ", ".join(f'{k}:{v}' for k, v in metrics.items())
            )
            if metrics['tot_loss'].value() < best:
                best = metrics['tot_loss'].value()
                self.save_sdf(batch)
            self.scheduler.step()

class SDF_T:
    def __init__(self, sdf=None, D=None, msg=None, sdf_checkpoint=None):
        self.sdf = sdf
        if sdf is None:
            self.sdf = SDF(sdf_checkpoint=sdf_checkpoint)
        self.D = D
        if D is None:
            self.D = global_D
        if msg is None:
            self.msg = torch.asarray(np.random.choice([-1, 1], size=(V().cfg.msg_length, ))).cuda(V().cfg.device)
        else:
            self.msg = torch.as_tensor(msg).view(V().cfg.msg_length, ).to(V().cfg.device)
        
        self.points_batch_size = V().cfg.points_batch_size
        STEPS = self.linspace_steps = V().cfg.linspace_steps
        self.thetas = torch.linspace(0, torch.pi, STEPS + 1).cuda(V().cfg.device)
        self.phis = torch.linspace(-torch.pi, torch.pi, STEPS + 1).cuda(V().cfg.device)
    
    def get_msg(self, y):
        msg = self.msg
        y_sph = cartesian2spherical(y)
        y_theta_i = torch.searchsorted(self.thetas, y_sph[..., 1], right=True).clamp(0, self.linspace_steps) - 1
        y_phi_i = torch.searchsorted(self.phis, y_sph[..., 2], right=True).clamp(0, self.linspace_steps) - 1
        msg_idx = (y_theta_i * self.linspace_steps + y_phi_i) % msg.shape[0]
        return msg[msg_idx].detach(), msg_idx

    def T_forward(self, y, direction, create_graph=True):
        _, g = self.F_forward(y, create_graph=create_graph)
        g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return y + g * self.D(y) * direction
    
    def F_forward(self, x, with_grad=False, create_graph=False):
        return self.sdf(x, with_grad=with_grad, create_graph=create_graph)

    def __call__(self, x, with_grad=False, create_graph=False):
        """
        G(x)=F(T^{-1}(x))
        (1) y = T^{-1}(x) with grad
        (2) G(x)=F(y) no grad
        """
        x = x.to(V().cfg.device)
        direction = self.get_msg(x)[0]
        y = batched_T_reverse(self.T_forward, x, direction)
        return self.forward_with_y(x, y, with_grad=with_grad, create_graph=create_graph)
        
    def forward_with_y(self, x, y, with_grad=False, create_graph=False):
        """given (x, y), calculate G(x)=F(T(y)), G'(x)
        G(x)=F(T^{-1}(x))
        y=T^{-1}(x)
        (1) G(x)=F(y)
        """
        if create_graph: with_grad=True
        if with_grad:
            x, y = x.to(V().cfg.device), y.to(V().cfg.device)
            if y.is_leaf:
                y.requires_grad = True
            F, Fy = self.F_forward(y, create_graph=True)
            F2y2, status = jacobian(Fy, y, create_graph=False)
            ans = self.D(y) * F2y2 + torch.eye(Fy.shape[1], device=Fy.device).unsqueeze(0)
            yx = torch.linalg.inv(ans)
            Fx = torch.matmul(yx, Fy.unsqueeze(-1)).squeeze(-1)
        else:
            F, Fx = [], []
            split_x = torch.split(x, self.points_batch_size, dim=0)
            split_y = torch.split(y, self.points_batch_size, dim=0)
            for xi, yi in zip(split_x, split_y):
                xi, yi = xi.to(V().cfg.device), yi.to(V().cfg.device)
                yi.requires_grad = True
                Fi, Fyi = self.F_forward(yi, create_graph=True)
                F2y2i, status = jacobian(Fyi, yi, create_graph=False)
                ans = self.D(y) * F2y2i + torch.eye(Fyi.shape[1], device=Fyi.device).unsqueeze(0)
                yx = torch.linalg.inv(ans)
                Fxi = torch.matmul(yx, Fyi.unsqueeze(-1)).squeeze(-1)
                F.append(Fi.detach())
                Fx.append(Fxi.detach())
            F = torch.concat(F, dim=0)
            Fx = torch.concat(Fx, dim=0)
        return F, Fx

class SDF_WM:
    def __init__(self, msg=None, sdf_checkpoint=None, wm_checkpoint=None, sdf=None):
        self.msg = msg
        if msg is None:
            self.msg = torch.asarray(np.random.choice([-1, 1], size=(V().cfg.msg_length, ))).cuda(V().cfg.device)
        
        self.sdf_t = SDF_T(msg=self.msg, sdf_checkpoint=sdf_checkpoint, sdf=sdf)
        self.sdf = self.sdf_t.sdf

        if wm_checkpoint:
            self.batch = checkpoint = torch.load(wm_checkpoint)
            self.msg = checkpoint['msg'].to(V().cfg.device)
            sdf = V().cfg.sdf().cuda(V().cfg.device)
            sdf.load_state_dict(checkpoint['sdf'])
            self.sdf_wm = SDF(sdf=sdf)
            self.linspace_steps = STEPS = checkpoint["linspace_steps"]
        else:
            self.sdf_wm = SDF(sdf_checkpoint=wm_checkpoint)
            self.linspace_steps = STEPS = V().cfg.linspace_steps

        self.thetas = torch.linspace(0, torch.pi, STEPS + 1).cuda(V().cfg.device)
        self.phis = torch.linspace(-torch.pi, torch.pi, STEPS + 1).cuda(V().cfg.device)

    def get_msg(self, y):
        msg = self.msg
        y_sph = cartesian2spherical(y)
        y_theta_i = torch.searchsorted(self.thetas, y_sph[..., 1], right=True).clamp(0, self.linspace_steps) - 1
        y_phi_i = torch.searchsorted(self.phis, y_sph[..., 2], right=True).clamp(0, self.linspace_steps) - 1
        msg_idx = (y_theta_i * self.linspace_steps + y_phi_i) % msg.shape[0]
        return msg[msg_idx].detach(), msg_idx
            

    def prepare_batch(self):
        nsample = 500000
        # nsample = 10
        y, _, _ = get_surf_pcl(self.sdf, nsample, thr=1e-4)
        direction = self.sdf_t.get_msg(y)[0].view(y.shape[0], 1)
        x = self.sdf_t.T_forward(y, direction, create_graph=False)
        on_surface_sdf, on_surface_normal = self.sdf_t.forward_with_y(x, y)
        on_surface_x = x
        on_surface_normal = on_surface_normal / on_surface_normal.norm(dim=-1, keepdim=True)

        if False:
            def ori(points):
                pointsf = torch.from_numpy(points).cuda(V().cfg.device).float()
                values = - self.sdf(pointsf)[0][:, 0].double().cpu().numpy()
                return values
            xyz, faces = mise_plot(ori, 32, 2)
            xyz, faces = xyz.astype(np.float32), faces.astype(np.int32)

            p = pyvista.Plotter()
            mesh = to_pyvista(xyz, faces)
            moved_mesh = pyvista.PolyData(on_surface_x.detach().cpu().numpy())
            moved_mesh["Normals"] = on_surface_normal.detach().cpu().numpy()

            p.add_mesh(mesh)
            p.add_mesh(moved_mesh)
            p.add_mesh(moved_mesh.glyph(geom=pyvista.Arrow(scale=0.01), orient="Normals"), color="black")
            p.show()
            return
        off_surface_x = torch.FloatTensor(np.random.uniform(-1, 1, size=(nsample, 3))).to(V().cfg.device)
        off_surface_sdf, _ = self.sdf_t(off_surface_x)
        off_surface_normal = torch.ones_like(off_surface_x) * -1

        sdfs = torch.concat((on_surface_sdf, off_surface_sdf), dim=0)
        points = torch.concat((on_surface_x, off_surface_x), dim=0)
        normals = torch.concat((on_surface_normal, off_surface_normal), dim=0)

        batch = {
            "name": f'{self.sdf.batch["name"]}.wm',
            "points": points,
            "sdfs": sdfs,
            "normals": normals,
            "msg": self.msg,
            "linspace_steps": V().cfg.linspace_steps
        }
        # torch.save({
        #     "name": f'{self.sdf.batch["name"]}.wm',
        #     "points": points,
        #     "sdfs": sdfs,
        #     "normals": normals,
        #     "msg": self.msg,
        #     "linspace_steps": V().cfg.linspace_steps
        # }, os.path.join(V().cfg.output_dir, V().cfg.task_name, f'{self.sdf.batch["name"]}.data'))
        return batch

    def train_batch(self, batch):
        assert "name" in batch and \
            "points" in batch and \
            "sdfs" in batch and \
            "normals" in batch and \
            "msg" in batch and \
            "linspace_steps" in batch
        self.batch = batch
        self.sdf_wm.train_batch(batch)
    
    def __call__(self, x, with_grad=False, create_graph=False):
        return self.sdf_wm(x, with_grad=with_grad, create_graph=create_graph)