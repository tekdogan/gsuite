# GNN BM (draft)

#### :information_source: This repository incorporates the material about in-progress benchmark suite for Graph Neural Networks.

### Before Compiling
`export PATH=/usr/local/cuda-9.1/bin:$PATH`  
`export CPATH=/usr/local/cuda-9.1/include:$CPATH`  
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.1/lib64/`  

### Compilation
`nvcc -lcublas -std=c++11 -arch=compute_61 cuBlasUtil.cu Data_Util.cu CU_GCN_MP.cu CU_SpMM_GCN.cu cudaDataLoader.cu -o cudaDataLoader.o`  

### Profile via NVPROF
`nvprof -f --analysis-metrics -o cudaDataLoader.nvprof ./cudaDataLoader.o --benchmark`  
