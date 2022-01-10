#include "linear.h"

void linear(float *src, int srcRows, int srcCols,
              float *out, int outRows, int outCols) {
  
  float *w, *d_src, *d_out;
  
  // allocate device memory for output
  cudaMalloc((void**) &d_out, outRows*outCols*sizeof(float));

  cudaMalloc((void**) &d_src, srcRows*srcCols*sizeof(float));
  cudaMemcpy(d_src, src, srcRows*srcCols*sizeof(float), cudaMemcpyHostToDevice);
  
  // allocate device memory for weight
  cudaMalloc((void**) &w, srcCols*outCols*sizeof(float));

  float *h_w = (float*)calloc(srcCols*outCols, sizeof(float));
  memset(h_w, 1, srcCols*outCols*sizeof(float));
  cudaMemcpy(w, h_w, srcCols*outCols*sizeof(float), cudaMemcpyHostToDevice);


  // init weight matrix
  //initIdentityGPU<<<srcCols*outCols,1>>>(&w, srcCols, outCols);
  
  gpu_blas_mmul(w, d_src, d_out, srcRows, srcCols, outCols, false, false, 1.0, 0.0);
  
  cudaMemcpy(out,d_out,outRows*outCols*sizeof(float),cudaMemcpyDeviceToHost);
  
}
