#include "scatter_cuda.h"

#include "reducer.cuh"
//#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t, ReductionType REDUCE>
__global__ void
scatter_kernel(const float *src_data, float *out_data,
               int numOfNodes, int numOfFeatures, int numOfEdges) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx < numOfNodes*numOfFeatures*numOfEdges) {
    
    // get indices of the thread
    
    int64_t idx = (thread_idx / numOfEdges);
    
    int64_t index_info = thread_idx % (numOfFeatures*numOfEdges);
    
    int64_t id_r = (idx / numOfNodes);
    
    int64_t id_c = (id_r / numOfFeatures);

    if(( *(edgeIndex + edgeIndexSize + index_info) == id_c)) { // an incoming edge to node id_r
        Reducer<scalar_t, REDUCE>::atomic_write(out_data + id_r * numOfFeatures + id_c,
                                            src_data + id_r * numOfFeatures + id_c);
    }
  }
}

/*template <typename scalar_t>
__global__ void
scatter_arg_kernel(float *src_data, float *out_data, int64_t *arg_out_data,
                    int E, int K, int N, int numel) {

  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int b = thread_idx / (E * K);
  int e = (thread_idx / K) % E;
  int k = thread_idx % K;

  if (thread_idx < numel) {
    int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(
        thread_idx, index_info);
    int64_t idx = index_info.data[offset];

    if (src_data[thread_idx] == out_data[b * N * K + idx * K + k]) {
      arg_out_data[b * N * K + idx * K + k] = e;
    }
  }
}*/

std::tuple<float*, float*>
scatter_cuda(float *src, float *index, int64_t dim,
             std::string reduce, int numOfNodes, int numOfFeatures,
             int numOfEdges) {
  
  CHECK_CUDA(src);
  CHECK_CUDA(index);
  
  cudaSetDevice(0);
  
  float* out = calloc(edgeIndexSize * 2, sizeof(float));
  
  float* arg_out = calloc(edgeIndexSize * 2, sizeof(float));
  
  int64_t *arg_out_data = nullptr;
  
  
  // pick dim = 1
  auto B = numOfNodes; // mul of each dimension less than dim
  auto E = numOfFeatures; // size of dimension in dim
  auto K = 1; // mul of each dimension greater than dim
  auto N = numOfFeatures; // output size of dim


    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      //if (!optional_out.has_value())
      //  out.fill_(Reducer<scalar_t, REDUCE>::init());

      scatter_kernel<scalar_t, REDUCE>
          <<<BLOCKS(numOfNodes*numOfFeatures*numOfEdges), THREADS, 0, stream>>>(
              src_data, out_data, numOfNodes, numOfFeatures, numOfEdges);

      //if (!optional_out.has_value() && (REDUCE == MIN || REDUCE == MAX))
      //  out.masked_fill_(out == Reducer<scalar_t, REDUCE>::init(), (scalar_t)0);

      if (REDUCE == MIN || REDUCE == MAX)
        scatter_arg_kernel<scalar_t>
            <<<BLOCKS(numOfNodes), THREADS, 0, stream>>>(
                src_data, out_data, numOfNodes, numOfFeatures,
                numOfEdges);
    });
  });

  return std::make_tuple(out, arg_out);
}
