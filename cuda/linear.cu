#include "linear.h"

float* linear(float *src, int srcRows, int srcCols, int outRows, int outCols) {
  
  float *w, *d_src, *d_out;
  
  float *out;
  out = (float*)calloc(outRows*outCols, sizeof(float));

  // allocate device memory for output
  cudaError_t e = cudaMalloc((void**) &d_out, outRows*outCols*sizeof(float));

  const char* err = cudaGetErrorString(e);

  cudaMalloc((void**) &d_src, srcRows*srcCols*sizeof(float));
  e = cudaMemcpy(d_src, src, srcRows*srcCols*sizeof(float), cudaMemcpyHostToDevice);
  err = cudaGetErrorString(e);  


  // allocate device memory for weight
  cudaMalloc((void**) &w, srcCols*outCols*sizeof(float));

  float *h_w = (float*)calloc(srcCols*outCols, sizeof(float));
  memset(h_w, 1, srcCols*outCols*sizeof(float));
  e = cudaMemcpy(w, h_w, srcCols*outCols*sizeof(float), cudaMemcpyHostToDevice);
  err = cudaGetErrorString(e);

  // init weight matrix
  //initIdentityGPU<<<srcCols*outCols,1>>>(&w, srcCols, outCols);
  
  gpu_blas_mmul(w, d_src, d_out, srcRows, srcCols, outCols, false, false, 1.0, 0.0);

  cudaDeviceSynchronize();

  e = cudaMemcpy(out,d_out,outRows*outCols*sizeof(float),cudaMemcpyDeviceToHost);
  err = cudaGetErrorString(e);

  e = cudaFree(d_out);
  cudaFree(w);
  cudaFree(d_src);
  
  return out;
}
