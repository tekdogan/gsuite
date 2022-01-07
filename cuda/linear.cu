#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cublas_v2.h>
#include "cuBlasUtil.h"

void linear(float *src, int srcRows, int srcCols
              float *out, int outRows, int outCols,
               float *bias) {
  
  float *w;
  
  // allocate device memory for output
  //cudaMalloc(&y,outRows*outCols*sizeof(float));
  
  // allocate device memory for weight
  cudaMalloc(&w,srcCols*sizeof(float));
  
  // init weight matrix
  initIdentityGPU<<<srcCols/128,128>>>(&w, srcCols, 1);
  
  gpu_blas_mmul(w, src, out, srcRows, srcCols, outCols, false, false, 1.0, 0.0);
  
  //cudaMemcpy(out,y,outRows*outCols*sizeof(float),cudaMemcpyDeviceToHost);
  
}
