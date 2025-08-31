from typing import Any
import torch
import torch.nn as nn
from torch.autograd import Function

from funcwm import cu3d

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        """
        xyz: (B, N, 3)
        npoint: int
        """
        fps_inds = cu3d.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(fps_inds)
        return fps_inds

    @staticmethod
    def backward(xyz, a=None):
        return None, None

furthest_point_sample = FurthestPointSampling.apply

class BallQuery(Function):
    @staticmethod
    def forward(ctx, ref, query, radius, nsample):
        idx, num = cu3d.ball_query(ref, query, radius, nsample)
        ctx.mark_non_differentiable(idx, num)
        return idx, num
    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None

ball_query = BallQuery.apply

class KNNQuery(Function):
    @staticmethod
    def forward(ctx, ref, query, nsample):
        idx, dist = cu3d.knn_query(ref, query, nsample)
        ctx.mark_non_differentiable(idx, dist)
        return idx, dist
    @staticmethod
    def backward(ctx, a=None):
        return None, None, None

knn_query = KNNQuery.apply

class GraphNeighborQuery(Function):
    @staticmethod
    def forward(ctx, xyz, faces):
        idx, num = cu3d.graph_neighbor_query(xyz, faces)
        ctx.mark_non_differentiable(idx, num)
        return idx, num
    @staticmethod
    def backward(ctx, a=None):
        return None, None

graph_neighbor_query = GraphNeighborQuery.apply

class GatherOperation(Function):
    @staticmethod
    def forward(ctx, ref, idx, channel_first):
        N = ref.size(2) if channel_first else ref.size(1)
        ctx.for_backwards = (idx, N, channel_first)
        return cu3d.gather_points(ref, idx, channel_first)

    @staticmethod
    def backward(ctx, grad_out):
        idx, N, channel_first = ctx.for_backwards
        ref_grad = cu3d.gather_points_grad(grad_out.contiguous(), idx, N, channel_first)
        return ref_grad, None, None

gather_operation = GatherOperation.apply

class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, ref, idx, channel_first):
        N = ref.size(2) if channel_first else ref.size(1)
        ctx.for_backwards = (idx, N, channel_first)
        return cu3d.group_points(ref, idx, channel_first)

    @staticmethod
    def backward(ctx, grad_out):
        idx, N, channel_first = ctx.for_backwards
        grad_features = cu3d.group_points_grad(grad_out.contiguous(), idx, N, channel_first)
        return grad_features, None, None

grouping_operation = GroupingOperation.apply



if __name__ == "__main__":
    pass