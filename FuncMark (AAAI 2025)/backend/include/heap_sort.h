

#pragma once

#include "cuda_utils.h"

__device__ void heap_swap(const int idx1, const int idx2,
                     float *__restrict__ heap_value, int *__restrict__ heap_index) {
    float idx1_value = heap_value[idx1];
    int idx1_idx = heap_index[idx1];
    heap_value[idx1] = heap_value[idx2];
    heap_index[idx1] = heap_index[idx2];
    heap_value[idx2] = idx1_value;
    heap_index[idx2] = idx1_idx;
}

__device__ void heap_up(int cnt, float *__restrict__ heap_value, int *__restrict__ heap_index) {
    while (cnt) {
        if (heap_value[cnt] > heap_value[cnt >> 1]) {
            heap_swap(cnt, cnt >> 1, heap_value, heap_index);
        } else {
            break;
        }
        cnt >>= 1;
    }
}

__device__ void heap_down(int cnt, const int heap_size, float *__restrict__ heap_value, int *__restrict__ heap_index) {
    while ((cnt << 1) < heap_size) {
        int left = cnt << 1, right = cnt << 1 | 1;
        int target = (right >= heap_size || heap_value[left] > heap_value[right]) ? left : right;

        if (heap_value[cnt] <= heap_value[target]) {
            heap_swap(cnt, target, heap_value, heap_index);
        } else {
            break;
        }
        cnt = target;
    }
}

__device__ bool heap_insert(const float value, const int index,
                            const int cnt, const int max_heap_size,
                            float *__restrict__ heap_value,
                            int *__restrict__ heap_index) {
    if (cnt < max_heap_size) { // not full
        heap_value[cnt] = value;
        heap_index[cnt] = index;
        heap_up(cnt, heap_value, heap_index);
        return true;
    }
    if (value >= *heap_value) { // larger than heap top
        return false;
    }
    heap_swap(0, cnt - 1, heap_value, heap_index); // delete heap top element
    heap_down(0, cnt - 1, heap_value, heap_index); // down new top element
    heap_value[cnt - 1] = value; // insert new element
    heap_index[cnt - 1] = index;
    heap_up(cnt - 1, heap_value, heap_index);
    return false;
}