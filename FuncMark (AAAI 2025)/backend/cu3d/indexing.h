

#pragma once

#include <torch/extension.h>

#include "cuda_utils.h"

void group_points_wrapper(int B, int channel_size, int ref_nb, int index_nb, int nsample, bool channel_first,
                                 const float *ref, const int *idx,
                                 float *out);

void group_points_grad_wrapper(int B, int channel_size, int ref_nb, int index_nb, int nsample, bool channel_first,
                               const float *grad_out, const int* idx,
                               float *ref_grad);

void gather_points_wrapper(int B, int channel_size, int ref_nb, int index_nb, bool channel_first,
                           const float *ref, const int *idx,
                           float *out);

void gather_points_grad_wrapper(int B, int channel_size, int ref_nb, int index_nb, bool channel_first,
                                const float *ref, const int *idx,
                                float *ref_grad);

