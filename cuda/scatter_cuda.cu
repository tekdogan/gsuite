#include "scatter_cuda.h"

template <typename scalar_t, ReductionType REDUCE>
__global__ void
scatter_kernel(const float *src_data, const int *indices, float *out_data,
               int numOfRows, int numOfColumns, int indSize, int dim) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // TO DO: size of src_data
  if (thread_idx < indSize) {

    Reducer<scalar_t, REDUCE>::atomic_write(out_data + (int)*(indices + thread_idx),
                                            *(src_data + thread_idx));
  }
}


float* scatter_cuda(float *h_src, int *h_index, int64_t dim,
             std::string reduce, int indSize, int srcRows,
             int srcCols, int outRows, int outCols) {
    
  cudaSetDevice(0);
  
  float *d_src;
  cudaMalloc((void**) &d_src, srcRows*srcCols*sizeof(float));
  cudaMemcpy(d_src, h_src, srcRows*srcCols*sizeof(float), cudaMemcpyHostToDevice);

  int *d_index;
  cudaMalloc((void**) &d_index, indSize*sizeof(int));
  cudaMemcpy(d_index, h_index, indSize*sizeof(int), cudaMemcpyHostToDevice);  

  float *d_out;
  cudaMalloc((void**) &d_out, outRows*outCols*sizeof(float));
  

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      //if (!optional_out.has_value())
      //  out.fill_(Reducer<scalar_t, REDUCE>::init());

      scatter_kernel<float, REDUCE>
          <<<BLOCKS(srcRows*srcCols), THREADS>>>(
              d_src, d_index, d_out, srcRows, srcCols,
		indSize, dim);

    });

  float *h_out = (float*)calloc(outRows*outCols, sizeof(float));
  cudaMemcpy(h_out, d_out, outRows*outCols*sizeof(float), cudaMemcpyDeviceToHost);

  return h_out;
}
