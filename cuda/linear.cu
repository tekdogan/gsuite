#include "linear.h"

void linear(float *src, int srcRows, int srcCols,
              float *out, int outRows, int outCols) {
  
  float *w, *d_src, *d_out;
  
  // allocate device memory for output
  cudaMalloc(&d_out, outRows*outCols*sizeof(float));

  cudaMalloc(&d_src, srcRows*srcCols*sizeof(float));
  cudaMemcpy(d_src, src, srcRows*srcCols*sizeof(float), cudaMemcpyHostToDevice);
  
  // allocate device memory for weight
  cudaMalloc(&w, srcCols*outCols*sizeof(float));
  
  // init weight matrix
  initIdentityGPU<<<srcCols/128,128>>>(&w, srcCols, outCols);
  
  gpu_blas_mmul(w, d_src, d_out, srcRows, srcCols, outCols, false, false, 1.0, 0.0);
  
  //cudaMemcpy(out,y,outRows*outCols*sizeof(float),cudaMemcpyDeviceToHost);
  
}
