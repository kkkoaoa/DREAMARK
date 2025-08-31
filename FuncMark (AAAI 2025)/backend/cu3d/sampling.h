

#pragma once

#include "cuda_utils.h"

void furthest_point_sampling_wrapper(int B, int ref_nb, int nsample,
                                     const float *dataset, float *temp,
                                     int *idxs);
