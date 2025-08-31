

#include "nnquery.h"
#include "heap_sort.h"

__global__ void knn_query_kernel(int B,
                      int ref_nb,
                      int query_nb,
                      int nsample,
                      const float* ref,
                      const float* query,
                      float* knn_dist,
                      int* knn_index) {
    // printf("B=%d query_nb=%d\n", B, query_nb);
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        for (int q = threadIdx.x; q < query_nb; q += blockDim.x) {
            // printf("b=%d q=%d threadIdx.x=%d blockDim.x=%d query_nb=%d (b*query_nb+q)*3=%d query[]=%f\n", b, q, threadIdx.x, blockDim.x, query_nb, (b*query_nb+q)*3, query[(b*query_nb+q)*3+0]);
            float query_x = query[(b * query_nb + q) * 3 + 0];
            float query_y = query[(b * query_nb + q) * 3 + 1];
            float query_z = query[(b * query_nb + q) * 3 + 2];

            for (int r = 0, cnt = 0; r < ref_nb; ++r) {
                float x = ref[(b * ref_nb + r) * 3 + 0];
                float y = ref[(b * ref_nb + r) * 3 + 1];
                float z = ref[(b * ref_nb + r) * 3 + 2];
                // printf("q=%d, r=%d, query=(%f, %f, %f), ref=(%f, %f, %f)\n", q, r, query_x, query_y, query_z, x, y, z);
                float dist = (query_x - x) * (query_x - x) + (query_y - y) * (query_y - y) + (query_z - z) * (query_z - z);
                cnt += heap_insert(dist, r, cnt, nsample,
                                   knn_dist + (b * query_nb + q) * nsample,
                                   knn_index + (b * query_nb + q) * nsample);
            }
        }
    }
}

/**
 *
 * @param B
 * @param ref_nb
 * @param query_nb
 * @param nsample
 * @param ref       (B, ref_nb, 3)
 * @param query     (B, query_nb, 3)
 * @param knn_dist  (B, query_nb, nsample)
 * @param knn_index (B, query_nb, nsample)
 */
void knn_cuda_wrapper(int B,
                      int ref_nb,
                      int query_nb,
                      int nsample,
                      const float* ref,
                      const float* query,
                      float* knn_dist,
                      int* knn_index) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    knn_query_kernel<<<opt_n_threads(B), opt_n_threads(query_nb), 0, stream>>>(
            B, ref_nb, query_nb, nsample, ref, query, knn_dist, knn_index);
    CUDA_CHECK_ERRORS();
}


__global__ void ball_query_kernel(int B,
                                  int ref_nb,
                                  int query_nb,
                                  float radius,
                                  int nsample,
                                  const float *ref,
                                  const float *query,
                                  int *cnt,
                                  int *idx) {
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        for (int q = threadIdx.x; q < query_nb; q += blockDim.x) {
            float qx = query[(b * query_nb + q) * 3 + 0];
            float qy = query[(b * query_nb + q) * 3 + 1];
            float qz = query[(b * query_nb + q) * 3 + 2];
            for (int r = threadIdx.y; r < ref_nb; r += blockDim.y) {
                float rx = ref[(b * ref_nb + r) * 3 + 0];
                float ry = ref[(b * ref_nb + r) * 3 + 1];
                float rz = ref[(b * ref_nb + r) * 3 + 2];
                float d2 = (rx - qx) * (rx - qx) + (ry - qy) * (ry - qy) +
                        (rz - qz) * (rz - qz);
                if (d2 < radius * radius && cnt[b * query_nb + q] < nsample) {
                    int c = atomicAdd(cnt + b * query_nb + q, 1);
//                    printf("batch%d: q%d=(%f,%f,%f), r%d=(%f,%f,%f), d=%f, <R?=%d, cnt=%d\n", b, q, qx, qy, qz, r, rx, ry, rz, d2, d2 < radius * radius, c);
                    if (c < nsample) { // To prevent concurrency bug
                        idx[(b * query_nb + q) * nsample + c] = r;
                    }
                }
            }
        }
    }
}

/**
 *
 * @param B
 * @param ref_nb
 * @param query_nb
 * @param radius
 * @param nsample
 * @param ref      (B, ref_nb, 3)
 * @param query    (B, query_nb, 3)
 * @param cnt      (B, query_nb)
 * @param idx      (B, query_nb, nsample)
 */
void ball_query_wrapper(int B,
                        int ref_nb,
                        int query_nb,
                        float radius,
                        int nsample,
                        const float *ref,
                        const float *query,
                        int *cnt,
                        int *idx) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ball_query_kernel<<<opt_n_threads(B), opt_block_config(query_nb, ref_nb), 0, stream>>>(
            B, ref_nb, query_nb, radius, nsample, ref, query, cnt, idx);
    CUDA_CHECK_ERRORS();
}


/**
 *
 * @param B
 * @param N
 * @param M
 * @param nsample
 * @param header (B, N)
 * @param cnt (B)
 * @param v (B, M)
 * @param next (B, M)
 * @param vis (B, N)
 * @param idx (B, N, nsample)
 * @param num (B, N)
 */
__global__ void graph_neighbor_query_kernel(int B, int N, int M, int nsample,
                                            const int *__restrict__ header,
                                            const int *__restrict__ cnt,
                                            const int *__restrict__ v,
                                            const int *__restrict__ next,
                                            int *__restrict__ idx, int *__restrict__ num) {
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        for (int n = threadIdx.x; n < N; n += blockDim.x) {
            for (int i = header[b*N+n]; i!=-1; i=next[b*M+i]) {
                int p = v[b*M+i];
                int current = num[b*N+n]++;
                idx[(b*N+n)*nsample+current] = p;
            }
        }
    }
}

void graph_neighbor_query_wrapper(int B, int N, int M, int nsample,
                                  const int *__restrict__ header,
                                     const int *__restrict__ cnt,
                                     const int *__restrict__ v,
                                     const int *__restrict__ next,
                                     int *__restrict__ idx, int *__restrict__ num) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    graph_neighbor_query_kernel<<<B, opt_n_threads(N), 0, stream>>>(B, N, M, nsample,
                                                                    header, cnt, v, next, idx, num);
    CUDA_CHECK_ERRORS();
}