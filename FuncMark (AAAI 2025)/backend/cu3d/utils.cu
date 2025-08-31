
#include "utils.h"

__device__ bool check_exist(int from, int to,
                            int *__restrict__ header,
                            int *__restrict__ cnt,
                            int *__restrict__ v,
                            int *__restrict__ next) {
    for (int i = header[from]; i != -1; i = next[i]) {
        if (v[i] == to) {
            return true;
        }
    }
    return false;
}

__device__ void add_edge(int from, int to,
                         int *__restrict__ header,
                         int *__restrict__ cnt,
                         int *__restrict__ v,
                         int *__restrict__ next,
                         int *__restrict__ neighbor_num) {
    if (check_exist(from, to, header, cnt, v, next)) {
        return;
    }
    int c = cnt[0]++;
    v[c] = to;
    next[c] = header[from];
    header[from] = c;
    ++neighbor_num[from];
}

/**
 *
 * @param B batch size
 * @param N number of vertices
 * @param M number of faces
 * @param faces (B, M, 3)
 * @param header (B, N)
 * @param cnt (B)
 * @param v (B, 6M)
 * @param next (B, 6M)
 * @param neighbor_num (B, N)
 */
__global__ void build_graph_from_face_kernel(int B, int N, int M, const int *__restrict__ faces,
                                   int *__restrict__ header,
                                   int *__restrict__ cnt,
                                   int *__restrict__ v,
                                   int *__restrict__ next,
                                   int *__restrict__ neighbor_num) {
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        // has to be sequential execution
        for (int m = 0; m < M; ++m) {
            int x = faces[(b * M + m) * 3], y = faces[(b * M + m) * 3 + 1], z = faces[(b * M + m) * 3 + 2];
            if (x == -1) {
                break;
            }
            add_edge(x, y, header+b*N, cnt+b, v+b*6*M , next+b*6*M, neighbor_num+b*N);
            add_edge(y, x, header+b*N, cnt+b, v+b*6*M , next+b*6*M, neighbor_num+b*N);
            add_edge(x, z, header+b*N, cnt+b, v+b*6*M , next+b*6*M, neighbor_num+b*N);
            add_edge(z, x, header+b*N, cnt+b, v+b*6*M , next+b*6*M, neighbor_num+b*N);
            add_edge(y, z, header+b*N, cnt+b, v+b*6*M , next+b*6*M, neighbor_num+b*N);
            add_edge(z, y, header+b*N, cnt+b, v+b*6*M , next+b*6*M, neighbor_num+b*N);
        }
    }
}

void build_graph_from_face_wrapper(int B,
                                   int N,
                                   int M,
                                   const int *__restrict__ faces,
                                   int *__restrict__ header,
                                   int *__restrict__ cnt,
                                   int *__restrict__ v,
                                   int *__restrict__ next,
                                   int *__restrict__ neighbor_num) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    build_graph_from_face_kernel<<<B, 1, 0, stream>>>(B, N, M, faces, header, cnt, v, next, neighbor_num);

    CUDA_CHECK_ERRORS();
}