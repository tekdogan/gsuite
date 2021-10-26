# GNN BM (draft)

#### :information_source: This repository incorporates the material about in-progress benchmark suite for Graph Neural Networks.

### Before Compiling
`export PATH=/usr/local/cuda-8.0/bin:$PATH`  
`export CPATH=/usr/local/cuda-8.0/include:$CPATH`  
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64/`  

### Compilation
`nvcc -lcublas -std=c++11 -arch=compute_61 cuBlasUtil.cu Data_Util.cu CU_GCN_MP.cu CU_SpMM_GCN.cu cudaDataLoader.cu -o cudaDataLoader.o`  

### Profile via NVPROF
`nvprof -f --analysis-metrics -o cudaDataLoader.nvprof ./cudaDataLoader.o --benchmark`  

### lib compilation
`nvcc -lcublas -std=c++11 -c -arch=compute_61 cuBlasUtil.cu Data_Util.cu CU_GCN_MP.cu CU_SpMM_GCN.cu cudaDataLoader.cu --compiler-options -fPIC`  
`nvcc --shared -o libCU_SpMM_GCN.so cudaDataLoader.o cuBlasUtil.o Data_Util.o CU_SpMM_GCN.o --compiler-options -fPIC -std=c++11`  
