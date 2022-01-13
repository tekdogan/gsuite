#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cublas_v2.h>
#include "cuBlasUtil.h"
#include "Data_Util.h"

float* linear(float *src, int srcRows, int srcCols,
              float *out, int outRows, int outCols);
