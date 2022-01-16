#include "scatter_cuda.h"

template <typename scalar_t, ReductionType REDUCE>
__global__ void
scatter_kernel(const float *src_data, const int *indices, float *out_data,
               int numOfRows, int numOfColumns, int indSize, int dim) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int input_size = numOfRows*numOfColumns;
  // TO DO: size of src_data
  if (thread_idx < input_size) {

    int src_r = thread_idx / numOfColumns;
    int out_r = indices[src_r];

    int src_c = thread_idx % numOfColumns;
    int out_c = src_c;

//    printf("out data index: %d\n, out_r: %d, src_r: %d, indSize: %d\n", out_r*numOfColumns + out_c, out_r, src_r, indSize);
    const float* address = src_data + src_r*numOfColumns + src_c;
    
    float data =  *(address);


    Reducer<scalar_t, REDUCE>::atomic_write(out_data + out_r*numOfColumns + out_c,
                                            data);

  }
}


float* scatter_cuda(float *h_src, int *h_index, int64_t dim,
             std::string reduce, int indSize, int srcRows,
             int srcCols, int outRows, int outCols) {
    
  cudaSetDevice(0);
  
  float *d_src;
  cudaError_t e = cudaMalloc((void**) &d_src, srcRows*srcCols*sizeof(float));
  const char* err = cudaGetErrorString(e);

  e = cudaMemcpy(d_src, h_src, srcRows*srcCols*sizeof(float), cudaMemcpyHostToDevice);
  err = cudaGetErrorString(e);


  int *d_index;
  e = cudaMalloc((void**) &d_index, indSize*sizeof(int));
  err = cudaGetErrorString(e);
  

  e = cudaMemcpy(d_index, h_index, indSize*sizeof(int), cudaMemcpyHostToDevice);  
  err = cudaGetErrorString(e);


  float *d_out;
  e = cudaMalloc((void**) &d_out, outRows*outCols*sizeof(float));
  err = cudaGetErrorString(e);


 printf("out max size: %d\n", outRows*outCols);

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      //if (!optional_out.has_value())
      //  out.fill_(Reducer<scalar_t, REDUCE>::init());

      scatter_kernel<float, REDUCE>
          <<<BLOCKS(srcRows*srcCols), THREADS>>>(
              d_src, d_index, d_out, srcRows, srcCols,
		indSize, dim);

    });

  printf("debug scatter kernel launched\n");

  cudaDeviceSynchronize();
  float *h_out = (float*)calloc(outRows*outCols, sizeof(float));
  e = cudaMemcpy(h_out, d_out, outRows*outCols*sizeof(float), cudaMemcpyDeviceToHost);
  err = cudaGetErrorString(e);

  e = cudaFree(d_out);
  err = cudaGetErrorString(e);

  e = cudaFree(d_src);
  err = cudaGetErrorString(e);
  
  e = cudaFree(d_index);
  err = cudaGetErrorString(e);

  return h_out;
}
