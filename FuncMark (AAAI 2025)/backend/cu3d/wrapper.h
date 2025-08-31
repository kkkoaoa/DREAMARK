#pragma once

#include <torch/extension.h>
#include <vector>

#include "cuda_utils.h"
#include "utils.h"
#include "nnquery.h"
#include "sampling.h"
#include "indexing.h"

/**
 * utils.h
*/
std::vector<at::Tensor> build_graph_from_face(at::Tensor xyz, at::Tensor faces) {
    CHECK_CONTIGUOUS(faces);
    CHECK_CUDA(faces);
    CHECK_IS_INT(faces);

    at::cuda::CUDAGuard device_guard(xyz.device());

    const int B = xyz.size(0);
    const int N = xyz.size(1);
    const int M = faces.size(1);

    at::Tensor header = torch::full({B, N}, -1, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor cnt = torch::zeros({B}, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor v = torch::zeros({B, 6*M}, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor next = torch::full({B, 6*M}, -1, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor neighbor_num = torch::zeros({B, N}, at::device(xyz.device()).dtype(at::ScalarType::Int));

    build_graph_from_face_wrapper(B, N, M, faces.data_ptr<int>(), header.data_ptr<int>(),
                                  cnt.data_ptr<int>(), v.data_ptr<int>(), next.data_ptr<int>(), neighbor_num.data_ptr<int>());

    return std::vector<at::Tensor>{header, cnt, v, next, neighbor_num};
}

/**
 *
 * @param xyz (N, 3)
 * @param faces (M, 3)
 * @return
 */
std::vector<at::Tensor> find_closed_surface(at::Tensor xyz, at::Tensor faces) {
    CHECK_CONTIGUOUS(faces);
    CHECK_CUDA(faces);
    CHECK_IS_INT(faces);
    CHECK_CONTIGUOUS(xyz);
    CHECK_CUDA(xyz);
    CHECK_IS_FLOAT(xyz);

    
}

/**
 * nnquery.h
*/
std::vector<at::Tensor> knn_query(at::Tensor ref, at::Tensor query, int nsample) {
    CHECK_CONTIGUOUS(ref);
    CHECK_CONTIGUOUS(query);
    CHECK_CUDA(ref);
    CHECK_CUDA(query);
    CHECK_IS_FLOAT(ref);
    CHECK_IS_FLOAT(query);

    at::cuda::CUDAGuard device_guard(ref.device());

    const int B = ref.size(0);
    const int ref_nb = ref.size(1);
    const int query_nb = query.size(1);

    at::Tensor knn_dist = torch::zeros({B, query_nb, nsample}, at::device(ref.device()).dtype(at::ScalarType::Float));
    at::Tensor knn_index = torch::full({B, query_nb, nsample}, -1, at::device(ref.device()).dtype(at::ScalarType::Int));
    // DEBUG(ref.device(), query.device(), knn_dist.device(), knn_index.device());

    knn_cuda_wrapper(B, ref_nb, query_nb, nsample, ref.data_ptr<float>(), query.data_ptr<float>(),
                     knn_dist.data_ptr<float>(), knn_index.data_ptr<int>());

    return std::vector<at::Tensor>{knn_index, knn_dist.sqrt()};
}

std::vector<at::Tensor> ball_query(at::Tensor ref, at::Tensor query, const float radius, const int nsample) {
    CHECK_CONTIGUOUS(ref);
    CHECK_CONTIGUOUS(query);
    CHECK_IS_FLOAT(ref);
    CHECK_IS_FLOAT(query);
    CHECK_CUDA(ref);
    CHECK_CUDA(query);

    at::cuda::CUDAGuard device_guard(ref.device());

    const int B = ref.size(0);
    const int ref_nb = ref.size(1);
    const int query_nb = query.size(1);

    at::Tensor idx = torch::full({B, query_nb, nsample}, -1, at::device(ref.device()).dtype(at::ScalarType::Int));
    at::Tensor cnt = torch::zeros({B, query_nb}, at::device(ref.device()).dtype(at::ScalarType::Int));

    ball_query_wrapper(B, ref_nb, query_nb, radius, nsample,
                       ref.data_ptr<float>(), query.data_ptr<float>(), cnt.data_ptr<int>(), idx.data_ptr<int>());
    at::Tensor num = (idx!=-1).sum(-1);
    return std::vector<at::Tensor>{idx, cnt};
}

std::vector<at::Tensor> graph_neighbor_query(at::Tensor xyz, at::Tensor faces) {
    CHECK_CONTIGUOUS(xyz);
    CHECK_CONTIGUOUS(faces);
    CHECK_CUDA(xyz);
    CHECK_CUDA(faces);
    CHECK_IS_FLOAT(xyz);
    CHECK_IS_INT(faces);

    at::cuda::CUDAGuard device_guard(xyz.device());

    const int B = xyz.size(0);
    const int N = xyz.size(1);

    auto rtn = build_graph_from_face(xyz, faces);
    auto header = rtn[0];
    auto cnt = rtn[1];
    auto v = rtn[2];
    auto next = rtn[3];
    int largest_neighbor_num = rtn[4].max().item<int>();

    const int M = v.size(1);

    at::Tensor idx = torch::full({B, N, largest_neighbor_num}, -1, at::device(xyz.device()).dtype(at::ScalarType::Int));
    at::Tensor num = torch::zeros({B, N}, at::device(xyz.device()).dtype(at::ScalarType::Int));

    graph_neighbor_query_wrapper(B, N, M, largest_neighbor_num,
                                 header.data_ptr<int>(), cnt.data_ptr<int>(), v.data_ptr<int>(), next.data_ptr<int>(),
                                 idx.data_ptr<int>(), num.data_ptr<int>());
    return std::vector<at::Tensor>{idx, num};
//    at::Tensor kidx = k_neighbor_query(xyz, xyz, nsample);
//    return merge_idx(idx, kidx);
}

/**
 * indexing.h
*/

at::Tensor gather_points(at::Tensor ref, at::Tensor idx, const bool channel_first) {
    CHECK_CONTIGUOUS(ref);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(ref);
    CHECK_IS_INT(idx);
    CHECK_CUDA(ref);
    CHECK_CUDA(idx);

    at::cuda::CUDAGuard device_guard(ref.device());

    if (channel_first) {
        /**
         * ref: (B, channel_size, ref_nb)
         * idx: (B, index_nb)
         * out: (B, channel_size, index_nb)
         */
        const int B = ref.size(0);
        const int channel_size = ref.size(1);
        const int ref_nb = ref.size(2);
        const int index_nb = idx.size(1);

        at::Tensor out = torch::zeros({B, channel_size, index_nb}, at::device(ref.device()).dtype(at::ScalarType::Float));

        gather_points_wrapper(B, channel_size, ref_nb, index_nb, channel_first,
                              ref.data_ptr<float>(), idx.data_ptr<int>(), out.data_ptr<float>());
        return out;
    } else {
        /**
         * ref: (B, ref_nb, channel_size)
         * idx: (b, index_nb)
         * out: (B, index_nb, channel_size)
         */
        const int B = ref.size(0);
        const int channel_size = ref.size(2);
        const int ref_nb = ref.size(1);
        const int index_nb = idx.size(1);

        at::Tensor out = torch::zeros({B, index_nb, channel_size}, at::device(ref.device()).dtype(at::ScalarType::Float));

        gather_points_wrapper(B, channel_size, ref_nb, index_nb, channel_first,
                              ref.data_ptr<float>(), idx.data_ptr<int>(), out.data_ptr<float>());
        return out;
    }
}

at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int ref_nb, const bool channel_first) {
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);
    CHECK_CUDA(idx);
    CHECK_CUDA(grad_out);

    at::cuda::CUDAGuard device_guard(grad_out.device());

    if (channel_first) {
        /**
         * grad_out: (B, channel_size, index_nb)
         * idx: (B, index_nb)
         * ref_grad: (B, channel_size, ref_nb)
         */
        const int B = grad_out.size(0);
        const int channel_size = grad_out.size(1);
        const int index_nb = grad_out.size(2);

        at::Tensor ref_grad = torch::zeros({B, channel_size, ref_nb}, at::device(grad_out.device()).dtype(at::ScalarType::Float));

        gather_points_grad_wrapper(B, channel_size, ref_nb, index_nb, channel_first,
                                   grad_out.data_ptr<float>(), idx.data_ptr<int>(), ref_grad.data_ptr<float>());
        return ref_grad;
    } else {
        /**
         * grad_out: (B, index_nb, channel_size)
         * idx: (B, index_nb)
         * ref_grad: (B, ref_nb, channel_size)
         */
        const int B = grad_out.size(0);
        const int channel_size = grad_out.size(2);
        const int index_nb = grad_out.size(1);

        at::Tensor ref_grad = torch::zeros({B, ref_nb, channel_size}, at::device(grad_out.device()).dtype(at::ScalarType::Float));

        gather_points_grad_wrapper(B, channel_size, ref_nb, index_nb, channel_first,
                                   grad_out.data_ptr<float>(), idx.data_ptr<int>(), ref_grad.data_ptr<float>());
        return ref_grad;
    }
}


at::Tensor group_points(at::Tensor ref, at::Tensor idx, const bool channel_first) {
    CHECK_CONTIGUOUS(ref);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(ref);
    CHECK_IS_INT(idx);
    CHECK_CUDA(ref);
    CHECK_CUDA(idx);

    at::cuda::CUDAGuard device_guard(ref.device());

    if (channel_first) {
        /**
         * ref: (B, channel_size, ref_nb)
         * idx: (B, index_nb, nsample)
         * out: (B, channel_size, index_nb, nsample)
         */
        const int B = ref.size(0);
        const int channel_size = ref.size(1);
        const int ref_nb = ref.size(2);
        const int index_nb = idx.size(1);
        const int nsample = idx.size(2);
        at::Tensor output = torch::zeros({B, channel_size, index_nb, nsample},
                                         at::device(ref.device()).dtype(at::ScalarType::Float));

        group_points_wrapper(B, channel_size, ref_nb, index_nb, nsample, channel_first,
                             ref.data_ptr<float>(), idx.data_ptr<int>(), output.data_ptr<float>());
        return output;
    } else {
        /**
         * ref: (B, ref_nb, channel_size)
         * idx: (B, index_nb, nsample)
         * out: (B, index_nb, nsample, channel_size)
         */
        const int B = ref.size(0);
        const int channel_size = ref.size(2);
        const int ref_nb = ref.size(1);
        const int index_nb = idx.size(1);
        const int nsample = idx.size(2);
        at::Tensor output = torch::zeros({B, index_nb, nsample, channel_size},
                                         at::device(ref.device()).dtype(at::ScalarType::Float));

        group_points_wrapper(B, channel_size, ref_nb, index_nb, nsample, channel_first,
                             ref.data_ptr<float>(), idx.data_ptr<int>(), output.data_ptr<float>());
        return output;
    }
}

at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int ref_nb, const bool channel_first) {
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);
    CHECK_CUDA(grad_out);
    CHECK_CUDA(grad_out);

    at::cuda::CUDAGuard device_guard(grad_out.device());

    if (channel_first) {
        /**
         * grad_out: (B, channel_size, index_nb, nsample)
         * idx: (B, index_nb, nsample)
         * ref_grad: (B, channel_size, ref_nb)
         */
        const int B = grad_out.size(0);
        const int index_nb = idx.size(1);
        const int channel_size = grad_out.size(1);
        const int nsample = idx.size(2);
        at::Tensor ref_grad = torch::zeros({B, channel_size, ref_nb}, at::device(grad_out.device()).dtype(at::ScalarType::Float));

        group_points_grad_wrapper(B, channel_size, ref_nb, index_nb, nsample, channel_first,
                                  grad_out.data_ptr<float>(), idx.data_ptr<int>(),
                                  ref_grad.data_ptr<float>());

        return ref_grad;
    } else {
        /**
         * grad_out: (B, index_nb, nsample, channel_size)
         * idx: (B, index_nb, nsample)
         * ref_grad: (B, ref_nb, channel_size)
         */
        const int B = grad_out.size(0);
        const int index_nb = idx.size(1);
        const int channel_size = grad_out.size(3);
        const int nsample = idx.size(2);
        at::Tensor ref_grad = torch::zeros({B, ref_nb, channel_size}, at::device(grad_out.device()).dtype(at::ScalarType::Float));

        group_points_grad_wrapper(B, channel_size, ref_nb, index_nb, nsample, channel_first,
                                  grad_out.data_ptr<float>(), idx.data_ptr<int>(),
                                  ref_grad.data_ptr<float>());

        return ref_grad;
    }
}


/**
 * sampling.h
*/

at::Tensor furthest_point_sampling(at::Tensor ref, const int nsamples) {
    CHECK_CONTIGUOUS(ref);
    CHECK_IS_FLOAT(ref);
    CHECK_CUDA(ref);

    at::cuda::CUDAGuard device_guard(ref.device());

    at::Tensor output = torch::full({ref.size(0), nsamples}, -1,
                                    at::device(ref.device()).dtype(at::ScalarType::Int));

    at::Tensor tmp = torch::full({ref.size(0), ref.size(1)}, 1e10,
                                 at::device(ref.device()).dtype(at::ScalarType::Float));

    furthest_point_sampling_wrapper(
            ref.size(0), ref.size(1), nsamples, ref.data_ptr<float>(),
            tmp.data_ptr<float>(), output.data_ptr<int>());

    return output;
}

/**
 * others
 */

at::Tensor sdf_by_normal(at::Tensor ref, at::Tensor normal, at::Tensor query) {
    CHECK_CONTIGUOUS(ref);
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(normal);
    CHECK_CUDA(ref);
    CHECK_CUDA(normal);
    CHECK_CUDA(query);
    CHECK(ref.size(1)==normal.size(1));

    at::cuda::CUDAGuard device_guard(ref.device());

    // DEBUG(query[0][0]);
    std::vector<at::Tensor> knn_res = knn_query(ref, query, 1);
    at::Tensor knn_index = knn_res[0]; // (B, query_nb, 1)
    // DEBUG(knn_index[0][0][0], query[0][knn_index[0][0][0]]);
    at::Tensor _1st_neighbor_pos = group_points(
            ref, knn_index, false); // (B, query_nb, 1, 3)
    at::Tensor _1st_neighbor_normal = group_points(
            normal, knn_index, false); // (B, query_nb, 1, 3)
    at::Tensor out = (query.unsqueeze(-2) - _1st_neighbor_pos).matmul(_1st_neighbor_normal.permute({0, 1, 3, 2})).squeeze(-1).squeeze(-1);
    return out;
}
