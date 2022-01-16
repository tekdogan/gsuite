#include <cuda.h>
#include <iostream>

__global__ void indexSelectLargeIndex(float *src, int srcRows, int srcCols,
                    int dim, int* indices, int indSize, int dstSize,
                    float *out);

float* index_select(float *src, int srcRows, int srcCols,
                    int dim, int* indices, int indSize);
