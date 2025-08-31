#pragma once
#include <torch/extension.h>

void implicit_laplacian_smoothing_impl(at::Tensor vertices, at::Tensor faces, float alpha, at::Tensor new_vertices);