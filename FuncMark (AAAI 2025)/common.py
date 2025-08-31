import os
import torch
import pyvista
import open3d as o3d
import numpy as np
from matplotlib.colors import ListedColormap

from funcwm.mise import MISE
from funcwm.mcubes import marching_cubes
from funcwm.dualcontour import dual_contouring
from backend import *
from util import *

GREEN_COLOR = np.array([159/256, 221/256, 240/256, 1.0])
BLUE_COLOR = np.array([12 / 256, 0 / 256, 246 / 256, 1.0])
BLACK_COLOR = np.array([11 / 256, 11 / 256, 11 / 256, 1.0])
GREY_COLOR = np.array([189 / 256, 189 / 256, 189 / 256, 1.0])
YELLOW_COLOR = np.array([255 / 256, 247 / 256, 0 / 256, 1.0])
RED_COLOR = np.array([1.0, 0.0, 0.0, 1.0])
ORANGE_COLOR = np.array([162/256, 62/256, 1/256, 1.0])

def pc_normalize(pc):
    if isinstance(pc, torch.Tensor):
        centroid = torch.mean(pc, axis=0).view(1, 3)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc**2, axis=-1)))
        pc = pc / m
    else:
        centroid = np.mean(pc, axis=0, keepdims=True)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=-1)))
        pc = pc / m
    return pc, centroid, m

def to_iterable(u):
    try:
        iter(u)
    except TypeError:
        u = torch.asarray([u])
    return u

def batch_cuda(x, cuda=True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not is_batch(x):
        x = x.unsqueeze(0)
    return x.cuda(V().cfg.device).contiguous() if cuda else x
    
def unbatch_cpu(x, cpu=True):
    if is_batch(x):
        x = x[0]
    return x.detach().cpu() if cpu else x

def is_batch(x):
    return len(x.shape) == 3

def to_open3d(vertices, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    return mesh

def to_pyvista(vertices, faces):
    faces = np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=-1).astype(np.int32)
    return pyvista.PolyData(vertices, faces)

#####################################################################
#                              IO                                   #
#####################################################################

def read_path(path):
    if os.path.splitext(path)[-1] == ".off":
        mesh = o3d.io.read_triangle_mesh(path)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
    else:
        mesh = pyvista.PolyData(path)
        vertices = np.asarray(mesh.points, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        faces = faces[:, 1:]
    return vertices, faces

def visual_path(path):
    vertices, faces = read_path(path)
    visual_mesh(vertices, faces)

def visual_mesh(vertices, faces, screenshot=None):
    mesh = to_pyvista(vertices, faces)
    p = pyvista.Plotter(off_screen=screenshot is not None)
    p.add_mesh(mesh, color='silver')
    # plane_cpos = [(2.0461563558065166, 1.2295711885634752, -0.6773245274980584),
    # (0.03988218053979646, 0.0005021609785960945, -0.13670381495746936),
    # (-0.4164379888475902, 0.8364671396917341, 0.35623324614710405)]
    cpos=None
    
    cpos = p.show(screenshot=screenshot, return_cpos=True, cpos=cpos)
    print(cpos)

#####################################################################
#                            Metrics                                #
#####################################################################


class Avg:
    def __init__(self):
        self.sum = 0
        self.cnt = 0
    def update(self, x, n):
        self.sum+=x
        self.cnt+=n
    def value(self):
        if self.cnt == 0:
            return 0
        return self.sum/self.cnt
    def __str__(self):
        return str(self.value())
    def __add__(self, other):
        res = Avg()
        res.sum = self.sum + other.sum
        res.cnt = self.cnt + other.cnt
        return res

def chamfer_distance(x, y):
    """
    x: shape (B, N, 3)
    y: shape (B, M, 3)
    """
    N, M = x.shape[1], y.shape[1]
    x, y = x.cuda(V().cfg.device), y.cuda(V().cfg.device)
    _, y_to_x = knn_query(x, y, 1)
    _, x_to_y = knn_query(y, x, 1)
    return y_to_x.sum()/y.shape[1] + x_to_y.sum()/x.shape[1]

#####################################################################
#                            Others                                 #
#####################################################################

"""
    The result of marching_cube be normalized to [-1, 1]
"""
def extract_mesh(occ_hat, normals=None, method='mc'):
    threshold = 0.5
    n_x, n_y, n_z = occ_hat.shape
    T = np.log(threshold) - np.log(1. - threshold)
    occ_hat_padded = occ_hat
    if method=='mc':
        vertices, faces = marching_cubes(
            occ_hat_padded, T)
    else:
        vertices, faces = dual_contouring(
            occ_hat_padded, normals
        )
    # print(vertices.shape)
    vertices -= (n_x-1)/2
    vertices = vertices/(n_x-1)*2
    return vertices, faces

"""
    This mise_plot will query indices in [-1, 1]
"""
def mise_plot(surf, res, dep, method='mc'):
    # mesh_extractor = MISE(res, dep, 0)
    # points = mesh_extractor.query()
    # while points.shape[0] != 0:
    #     pointsf = points / mesh_extractor.resolution
    #     pointsf = 2 * (pointsf - 0.5)
    #     values, _ = surf(pointsf)
    #     mesh_extractor.update(points, values)
    #     points = mesh_extractor.query()
    # occ_hat = mesh_extractor.to_dense()
    # xyz, faces = extract_mesh(occ_hat, method=method)
    if method=='mc':
        mesh_extractor = MISE(res, dep, 0)
        points = mesh_extractor.query()
        while points.shape[0] != 0:
            pointsf = points / mesh_extractor.resolution
            pointsf = 2 * (pointsf - 0.5)
            values, _ = surf(pointsf)
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()
        occ_hat = mesh_extractor.to_dense()
        return extract_mesh(occ_hat, method=method)
    elif method=='dc':
        # surf(np.array([]))
        x = np.linspace(0, res<<dep, (res<<dep)+1)
        xx, yy, zz = np.meshgrid(x, x, x)
        points = np.stack([yy.reshape(-1), xx.reshape(-1), zz.reshape(-1)]).T
        pointsf = points / (res<<dep)
        pointsf = 2*(pointsf-0.5)
        values, normals = surf(pointsf)
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        dist = values.reshape((res<<dep)+1, (res<<dep)+1, (res<<dep)+1)
        normals = normals.reshape((res<<dep)+1, (res<<dep)+1, (res<<dep)+1, 3)
        return extract_mesh(dist, normals, method=method)



def mesh_from_surf(surf):
    def pred(points):
        pointsf = torch.from_numpy(points).cuda(V().cfg.device).float()
        return_value = surf(pointsf)
        values = -return_value[0][:, 0].double().cpu().numpy()
        normals = return_value[1].double().cpu().numpy()
        return values, normals
    return pred

def plot(surf, visual_normal=False, screenshot=None):
    xyz, faces = mise_plot(mesh_from_surf(surf), 64, 2)
    xyz, faces = xyz.astype(np.float32), faces.astype(np.int32)
    # gauss = np.random.normal(loc=0, scale=0.001, size=xyz.shape).astype(np.float32)
    # xyz = xyz + gauss

    bit_color = np.empty((1, 4))
    bit_color[0] = GREEN_COLOR
    bit_color = ListedColormap(bit_color)
    
    p = pyvista.Plotter(off_screen=screenshot is not None)
    mesh = to_pyvista(xyz, faces)
    # mesh['color'] = np.zeros((xyz.shape[0], ))
    p.add_mesh(mesh, 
        # smooth_shading=True,
        # show_edges=True, edge_color="#30474E",
        # style="wireframe",
        color="silver", 
        # scalars='color', cmap=bit_color, show_scalar_bar=False,
    )
    
    if visual_normal:
        xyz_cuda = torch.from_numpy(xyz).cuda(V().cfg.device).requires_grad_(True)
        _, normal = surf(xyz_cuda)
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        normal = normal.cpu().numpy()
        mesh["Normals"] = normal
        p.add_mesh(mesh.glyph(geom=pyvista.Arrow(scale=0.01), orient="Normals"), color="black")
    # cpos = [(-0.3942531156655075, 0.656478326109289, 0.5155466272165394),
    # (-0.16098009093959897, 0.2977413012921601, -0.3113711016200095),
    # (0.040356785026367054, 0.9207504070657766, -0.38805929674541045)]
    cpos = None
    cpos = p.show(screenshot=screenshot, return_cpos=True, cpos=cpos)
    print(cpos)
    return xyz

#####################################################################
#                            Others                                 #
#####################################################################

def rotation_A_t(xyz, theta, A):
    """
    xyz:   shape (..., N, 3)
    theta: shape (..., 1)
    A:     shape (..., 3)
    """
    c = torch.cos(theta)
    s = torch.sin(theta)
    A = A / A.norm(dim=-1, keepdim=True)
    R = torch.stack([
        torch.stack([c+(1-c)*A[...,0]*A[...,0], (1-c)*A[...,0]*A[...,1]-s*A[...,2], (1-c)*A[...,0]*A[...,2]+s*A[...,1]], dim=-1),
        torch.stack([(1-c)*A[...,0]*A[...,1]+s*A[...,2], c+(1-c)*A[...,1]*A[...,1], (1-c)*A[...,1]*A[...,2]-s*A[...,0]], dim=-1),
        torch.stack([(1-c)*A[...,0]*A[...,2]-s*A[...,1], (1-c)*A[...,1]*A[...,2]+s*A[...,0], c+(1-c)*A[...,2]*A[...,2]], dim=-1),
    ], dim=-2).squeeze(-3).cuda(V().cfg.device)
    return torch.matmul(R, xyz.transpose(-1, -2)).transpose(-1, -2), R
