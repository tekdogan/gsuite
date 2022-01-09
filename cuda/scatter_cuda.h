#pragma once

#include <iostream>
#include "reducer.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

float* scatter_cuda(float *h_src, int *h_index, int64_t dim,
             std::string reduce, int indSize, int srcRows,
             int srcCols);

template <typename scalar_t, ReductionType REDUCE>
__global__ void
scatter_kernel(const float *src_data, const *indices, float *out_data,
               int numOfRows, int numOfColumns, int indSize, int dim);

