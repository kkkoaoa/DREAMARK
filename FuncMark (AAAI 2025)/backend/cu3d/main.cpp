
#include <vector>
#include <iostream>

#include "cuda_utils.h"
#include "wrapper.h"

void test_knn_query(){
    const int B = 1;
    const int N = 1;
    const int M = 1;
    const int batch_idx = 0;
    const int nsample = 1;

    torch::Device device(torch::kCUDA, 1);
    at::Tensor ref = torch::randn({B, N, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().to(device);
    at::Tensor query = torch::randn({B, M, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().to(device);

    auto rtn = knn_query(ref, query, nsample);
    auto idx = rtn[0];
    auto dist = rtn[1];
//    DEBUG(ref[batch_idx]);
//    DEBUG(query[batch_idx]);
    sleep(1);
    // DEBUG(idx[batch_idx]);
//    DEBUG(dist[batch_idx]);
//    auto dist_M = torch::zeros({B, N, M}, at::device(at::kCPU).dtype(at::ScalarType::Float));
//    for (int b = 0; b < B; ++b) {
//        for (int n = 0; n < N; ++n) {
//            for (int m = 0; m < M; ++m) {
//                auto dist_vec = ref[b][n] - query[b][m];
//                auto v = dist_vec.norm(2).item<float>();
////                DEBUG(b, n, m, ref[b][n], query[b][m], ref[b][n] - query[b][m], v);
//                *dist_M[b][n][m].data_ptr<float>() = v;
//            }
//        }
//    }
//    DEBUG(dist_M[batch_idx].permute({1, 0}));
}

void test_group_points() {
    const int B = 4;
    const int N = 4;
    const int M = 6;
    const int batch_idx = 3;
    const int nsample = 3;

    at::Tensor ref = torch::randn({B, 3, N}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    at::Tensor idx = torch::randint(N, {B, M, nsample}, at::device(at::kCPU).dtype(at::ScalarType::Int)).clone().cuda();

    auto output = group_points(ref, idx, true);
    // DEBUG(ref[batch_idx]);
    // DEBUG(idx[batch_idx]);
    // DEBUG(output[batch_idx]);
}

void test_group_points_grad() {
    const int B = 4;
    const int N = 4;
    const int M = 6;
    const int batch_idx = 3;
    const int c = 0;
    const int channel_size = 3;
    const int nsample = 3;

    at::Tensor ref = torch::randn({B, N, channel_size}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    at::Tensor idx = torch::randint(N, {B, M, nsample}, at::device(at::kCPU).dtype(at::ScalarType::Int)).clone().cuda();
    at::Tensor grad_out = torch::randn({B, M, nsample, channel_size}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    auto ref_grad = group_points_grad(grad_out, idx, N, false);
    // DEBUG(grad_out[batch_idx]);
    // DEBUG(idx[batch_idx]);
    // DEBUG(ref_grad[batch_idx]);
}

void test_gather_points() {
    const int B = 4;
    const int ref_nb = 4;
    const int index_nb = 6;
    const int channel_size = 3;
    const int b = 0;
    const int c = 0;

    at::Tensor ref = torch::randn({B, channel_size, ref_nb}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    at::Tensor idx = torch::randint(ref_nb, {B, index_nb}, at::device(at::kCPU).dtype(at::ScalarType::Int)).clone().cuda();
//    at::Tensor grad_out = torch::randn({B, index_nb, channel_size}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    auto out = gather_points(ref, idx, true);
    // DEBUG(ref[b]);
    // DEBUG(idx[b]);
    // DEBUG(out[b]);
}

void test_gather_points_grad() {
    const int B = 4;
    const int ref_nb = 4;
    const int index_nb = 6;
    const int channel_size = 3;
    const int b = 0;
    const int c = 0;

    at::Tensor ref = torch::randn({B, channel_size, ref_nb}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    at::Tensor idx = torch::randint(ref_nb, {B, index_nb}, at::device(at::kCPU).dtype(at::ScalarType::Int)).clone().cuda();
    at::Tensor grad_out = torch::randn({B, channel_size, index_nb}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    auto ref_grad = gather_points_grad(grad_out, idx, ref_nb, true);
    // DEBUG(grad_out[b]);
    // DEBUG(idx[b]);
    // DEBUG(ref_grad[b]);
}

void test_sdf_by_normal() {
    const int B = 4;
    const int N = 8;
    const int M = 6;
    const int batch_idx = 3;
    const int nsample = 3;

    at::Tensor ref = torch::randn({B, N, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    at::Tensor normal = torch::randn({B, N, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    at::Tensor query = torch::randn({B, M, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();

    auto rtn = sdf_by_normal(ref, normal, query);
    sleep(1);
    // DEBUG(ref[batch_idx]);
    // DEBUG(normal[batch_idx]);
    // DEBUG(query[batch_idx]);
    // DEBUG(rtn[batch_idx]);
}

void test_ball_query() {
    const int B = 4;
    const int N = 32000;
    const int M = 6;
    const int batch_idx = 3;
    const int nsample = 9;

    at::Tensor ref = torch::randn({B, N, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    at::Tensor query = torch::randn({B, M, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).clone().cuda();
    auto rtn = ball_query(ref, ref, 0.4, nsample);
    auto idx = rtn[0];
    auto num = rtn[1];
//    DEBUG(ref[batch_idx]);
//    DEBUG(query[batch_idx]);
    // DEBUG(idx[batch_idx]);
    // DEBUG((num[batch_idx] > nsample).sum());
}

void test_build_graph_from_face() {
    const int B = 4;
    const int N = 6;
    const int M = 6;
    const int batch_idx = 3;
    const int nsample = 9;
    at::Tensor xyz = torch::randn({B, N, 3}, at::device(at::kCUDA).dtype(at::ScalarType::Float));
    at::Tensor faces = torch::randint(N, {B, M, 3}, at::device(at::kCUDA).dtype(at::ScalarType::Int));
    DEBUG(faces[batch_idx]);
    auto rtn = build_graph_from_face(xyz, faces);
    auto header = rtn[0][batch_idx];
    auto cnt = rtn[1][batch_idx];
    auto v = rtn[2][batch_idx];
    auto next = rtn[3][batch_idx];
    auto neighbor_num = rtn[4][batch_idx];
    for (int i = 0; i < N; ++i) {
        for (int j = header[i].item<int>(); j!=-1; j=next[j].item<int>()) {
            int p = v[j].item<int>();
            DEBUG(i, p);
        }
    }
    DEBUG(neighbor_num);
}

void test_graph_neighbor_query() {
    const int B = 4;
    const int N = 6;
    const int M = 6;
    const int batch_idx = 3;
    const int nsample = 9;
    at::Tensor xyz = torch::randn({B, N, 3}, at::device(at::kCUDA).dtype(at::ScalarType::Float));
    at::Tensor faces = torch::randint(N, {B, M, 3}, at::device(at::kCUDA).dtype(at::ScalarType::Int));
    DEBUG(faces[batch_idx]);
    auto rtn = graph_neighbor_query(xyz, faces);
    auto idx = rtn[0];
    auto num = rtn[1];
    DEBUG(idx);
    DEBUG(num[batch_idx]);
}

int main() {
    test_graph_neighbor_query();
    return 0;
}