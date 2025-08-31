
#include "indexing.h"

__global__ void group_points_kernel(int B, int channel_size, int ref_nb, int index_nb, int nsample, bool channel_first,
                                    const float *__restrict__ ref,
                                    const int *__restrict__ idx,
                                    float *__restrict__ out) {
    if (channel_first) {
        /**
         * ref: (B, channel_size, ref_nb)
         * idx: (B, index_nb, nsample)
         * out: (B, channel_size, index_nb, nsample)
         */
        for (int b = blockIdx.x; b < B; b += gridDim.x) {
            for (int i = threadIdx.x; i < index_nb; i += blockDim.x) {
                for (int c = threadIdx.y; c < channel_size; c += blockDim.y) {
                    for (int k = 0; k < nsample; ++k) {
                        int r = idx[(b * index_nb + i) * nsample + k]; // r = idx[b, i, k]
                        out[ ((b * channel_size + c) * index_nb + i) * nsample + k ] = \
                            r!=-1? ref[ (b * channel_size + c) * ref_nb + r ] : 0; // out[b, c, i, k] = ref[b, c, r]
                    }
                }
            }
        }
    } else {
        /**
         * ref: (B, ref_nb, channel_size)
         * idx: (B, index_nb, nsample)
         * out: (B, index_nb, nsample, channel_size)
         */
        for (int b = blockIdx.x; b < B; b += gridDim.x) {
            for (int i = threadIdx.x; i < index_nb; i += blockDim.x) {
                for (int c = threadIdx.y; c < channel_size; c += blockDim.y) {
                    for (int k = 0; k < nsample; ++k) {
                        int r = idx[(b * index_nb + i) * nsample + k]; // r = idx[b, i, k]
                        out[ ((b * index_nb + i) * nsample + k) * channel_size + c ] = \
                            r!=-1? ref[ (b * ref_nb + r) * channel_size + c ] : 0; // out[b, i, k, c] = ref[b, r, c]
                    }
                }
            }
        }
    }
}


void group_points_wrapper(int B, int channel_size, int ref_nb, int index_nb, int nsample, bool channel_first,
                                 const float *ref, const int *idx,
                                 float *out) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    group_points_kernel<<<opt_n_threads(B), opt_block_config(index_nb, channel_size), 0, stream>>>(
            B, channel_size, ref_nb, index_nb, nsample, channel_first,
            ref, idx, out);

    CUDA_CHECK_ERRORS();
}


__global__ void group_points_grad_kernel(int B, int channel_size, int ref_nb, int index_nb, int nsample, bool channel_first,
                                         const float *grad_out, const int* idx,
                                         float *ref_grad) {
    if (channel_first) {
        /**
         * grad_out: (B, channel_size, index_nb, nsample)
         * idx: (B, index_nb, nsample)
         * ref_grad: (B, channel_size, ref_nb)
         */
        for (int b = blockIdx.x; b < B; b += gridDim.x) {
            for (int i = threadIdx.x; i < index_nb; i += blockDim.x) {
                for (int c = threadIdx.y; c < channel_size; c += blockDim.y) {
                    for (int k = 0; k < nsample; ++k) {
                        int r = idx[(b * index_nb + i) * nsample + k]; // r = idx[b, i, k]
//                        printf("idx[%d,%d,%d]=%d\t grad_out[%d,%d,%d,%d]=%f\n", b, i, k, r, b, c, i, k, grad_out[((b * channel_size + c) * index_nb + i) * nsample + k]);
                        if (r != -1) {
                            atomicAdd(ref_grad + (b * channel_size + c) * ref_nb + r, // ref_grad[b, c, r]
                                      grad_out[((b * channel_size + c) * index_nb + i) * nsample + k] // grad_out[b, c, i, k]
                            );
                        }
                    }
                }
            }
        }
    } else {
        /**
         * grad_out: (B, index_nb, nsample, channel_size)
         * idx: (B, index_nb, nsample)
         * ref_grad: (B, ref_nb, channel_size)
         */
        for (int b = blockIdx.x; b < B; b += gridDim.x) {
            for (int i = threadIdx.x; i < index_nb; i += blockDim.x) {
                for (int c = threadIdx.y; c < channel_size; c += blockDim.y) {
                    for (int k = 0; k < nsample; ++k) {
                        int r = idx[(b * index_nb + i) * nsample + k]; // r = idx[b, i, k]
                        if (r != -1) {
                            atomicAdd(ref_grad + (b * ref_nb + r) * channel_size + c, // ref_grad[b, r, c]
                                      grad_out[((b * index_nb + i) * nsample + k) * channel_size + c] // grad_out[b, i, k, c]
                            );
                        }
                    }
                }
            }
        }
    }
}


void group_points_grad_wrapper(int B, int channel_size, int ref_nb, int index_nb, int nsample, bool channel_first,
                               const float *grad_out, const int* idx,
                               float *ref_grad) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    group_points_grad_kernel<<<opt_n_threads(B), opt_block_config(index_nb, channel_size), 0, stream>>>(
            B, channel_size, ref_nb, index_nb, nsample, channel_first,
            grad_out, idx, ref_grad);
    CUDA_CHECK_ERRORS();
}


__global__ void gather_points_kernel(int B, int channel_size, int ref_nb, int index_nb, bool channel_first,
                          const float *ref, const int *idx,
                          float *out) {
    if (channel_first) {
        /**
         * ref: (B, channel_size, ref_nb)
         * idx: (B, index_nb)
         * out: (B, channel_size, index_nb)
         */
        for (int b = blockIdx.x; b < B; b += gridDim.x) {
            for (int i = threadIdx.x; i < index_nb; i += blockDim.x) {
                for (int c = threadIdx.y; c < channel_size; c += blockDim.y) {
                    int r = idx[b * index_nb + i]; // r = idx[b, i]
                    out[(b * channel_size + c) * index_nb + i] = \
                            (r != -1) ? ref[(b * channel_size + c) * ref_nb + r]: 0; // out[b, c, i] = ref[b, c, r]
                }
            }
        }
    } else {
        /**
         * ref: (B, ref_nb, channel_size)
         * idx: (B, index_nb)
         * out: (B, index_nb, channel_size)
         */
        for (int b = blockIdx.x; b < B; b += gridDim.x) {
            for (int i = threadIdx.x; i < index_nb; i += blockDim.x) {
                for (int c = threadIdx.y; c < channel_size; c += blockDim.y) {
                    int r = idx[b * index_nb + i]; // r = idx[b, i]
//                    printf("idx[%d,%d]=%d\t ref[%d,%d,%d]=%f\n", b, i, r, b, r, c, ref[(b * ref_nb + r) * channel_size + c]);
                    out[(b * index_nb + i) * channel_size + c] = \
                        (r != -1) ? ref[(b * ref_nb + r) * channel_size + c]: 0; // out[b, i, c] = ref[b, r, c]
                }
            }
        }
    }
}

void gather_points_wrapper(int B, int channel_size, int ref_nb, int index_nb, bool channel_first,
                           const float *ref, const int *idx,
                           float *out) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gather_points_kernel<<<opt_n_threads(B), opt_block_config(index_nb, channel_size), 0, stream>>>(
            B, channel_size, ref_nb, index_nb, channel_first,
            ref, idx, out);
    CUDA_CHECK_ERRORS();
}

__global__ void gather_points_grad_kernel(int B, int channel_size, int ref_nb, int index_nb, bool channel_first,
                                   const float *grad_out, const int* idx,
                                   float *ref_grad) {
    if (channel_first) {
        /**
         * grad_out: (B, channel_size, index_nb)
         * idx: (B, index_nb)
         * ref_grad: (B, channel_size, ref_nb)
         */
        for (int b = blockIdx.x; b < B; b += gridDim.x) {
            for (int i = threadIdx.x; i < index_nb; i += blockDim.x) {
                for (int c = threadIdx.y; c < channel_size; c += blockDim.y) {
                    int r = idx[b * index_nb + i]; // r = idx[b, i]
                    if (r != -1) {
                        atomicAdd(ref_grad + (b * channel_size + c) * ref_nb + r, // ref_grad[b, c, r]
                                  grad_out[(b * channel_size + c) * index_nb + i]); // grad_out[b, c, i]
                    }
                }
            }
        }
    } else {
        /**
         * grad_out: (B, index_nb, channel_size)
         * idx: (B, index_nb)
         * ref_grad: (B, ref_nb, channel_size)
         */
        for (int b = blockIdx.x; b < B; b += gridDim.x) {
            for (int i = threadIdx.x; i < index_nb; i += blockDim.x) {
                for (int c = threadIdx.y; c < channel_size; c += blockDim.y) {
                    int r = idx[b * index_nb + i]; // r = idx[b, i]
                    if (r != -1){
                        atomicAdd(ref_grad + (b * ref_nb + r) * channel_size + c, // ref_grad[b, r, c]
                                  grad_out[(b * index_nb + i) * channel_size + c]); // grad_out[b, i, c]
                    }
                }
            }
        }
    }
}

void gather_points_grad_wrapper(int B, int channel_size, int ref_nb, int index_nb, bool channel_first,
                                const float *grad_out, const int* idx,
                                float *ref_grad) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    gather_points_grad_kernel<<<opt_n_threads(B), opt_block_config(index_nb, channel_size), 0, stream>>>(
            B, channel_size, ref_nb, index_nb, channel_first,
            grad_out, idx, ref_grad);
    CUDA_CHECK_ERRORS();
}