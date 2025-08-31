
#pragma once

#include <torch/extension.h>

#include "cuda_utils.h"

void build_graph_from_face_wrapper(int B,
                                   int N,
                                   int M,
                                   const int *__restrict__ faces,
                                   int *__restrict__ header,
                                   int *__restrict__ cnt,
                                   int *__restrict__ v,
                                   int *__restrict__ next,
                                   int *__restrict__ neighbor_num);

