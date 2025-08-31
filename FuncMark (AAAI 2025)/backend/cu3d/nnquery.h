

#pragma once

#include <torch/extension.h>

#include "cuda_utils.h"
#include "utils.h"


void knn_cuda_wrapper(int B,
                      int ref_nb,
                      int query_nb,
                      int nsample,
                      const float* ref,
                      const float* query,
                      float* knn_dist,
                      int* knn_index);

void ball_query_wrapper(int B,
                        int ref_nb,
                        int query_nb,
                        float radius,
                        int nsample,
                        const float *ref,
                        const float *query,
                        int *cnt,
                        int *idx);

void graph_neighbor_query_wrapper(int B, int N, int M, int nsample,
                                  const int *__restrict__ header,
                                  const int *__restrict__ cnt,
                                  const int *__restrict__ v,
                                  const int *__restrict__ next,
                                  int *__restrict__ idx, int *__restrict__ num);
