#pragma once
#include <torch/extension.h>

std::vector<at::Tensor> remesh_impl(at::Tensor xyz, at::Tensor triangles, double target_edge_length, unsigned int nb_iter);