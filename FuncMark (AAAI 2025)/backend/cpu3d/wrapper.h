
#include "smoothing.h"
#include "remesh.h"

#include "cuda_utils.h"

at::Tensor implicit_laplacian_smoothing(at::Tensor xyz, at::Tensor faces, float alpha) {
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(faces);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_INT(faces);

    const int N = xyz.size(0);
    const int M = faces.size(0);

    at::Tensor new_vertices = torch::zeros({N, 3}, at::device(xyz.device()).dtype(at::ScalarType::Float));
    if (xyz.is_cuda()) { //gpu
        TORCH_CHECK(false, "gpu not supported");
    } else {
        implicit_laplacian_smoothing_impl(xyz, faces, alpha, new_vertices);
    }
    return new_vertices;
}

void remesh(at::Tensor xyz, at::Tensor faces, double target_edge_length, unsigned int nb_iter) {
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(faces);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_INT(faces);

    if (xyz.is_cuda()) {
        TORCH_CHECK(false, "gpu not supported");
    } else {
        remesh_impl(xyz, faces, target_edge_length, nb_iter);
    }
}