# GNN BM (draft)

#### :information_source: This repository incorporates the material about in-progress benchmark suite for Graph Neural Networks.

### UI Parameters
`--config`: (_mandatory_) local configuration file which includes default hyperparameters  
`--gpu-id` : (_optional_) device id  
`--model` : (_optional_) GNN model  
`--dataset` : (_optional_) dataset to be processed  

### Before Compiling
`export PATH=/usr/local/cuda-8.0/bin:$PATH`  
`export CPATH=/usr/local/cuda-8.0/include:$CPATH`  
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/usr/local/cuda-8.0/lib64/`  

### Compilation (deprecated, better use makefile)
_in cuda dir, compile codes using the command line below_    
`nvcc -lcublas -std=c++11 -arch=compute_61 cuBlasUtil.cu Data_Util.cu CU_GCN_MP.cu CU_SpMM_GCN.cu CU_SAG_WL.cu cudaDataLoader.cu main.cpp -o cudaDataLoader.o`  

_then call executable with a parameter_  
`./cudaDataLoader.o 2`

### Profile via NVPROF
`nvprof -f --analysis-metrics -o cudaDataLoader.nvprof ./cudaDataLoader.o --benchmark`  

### lib compilation
`nvcc -lcublas -std=c++11 -c -arch=compute_61 cuBlasUtil.cu Data_Util.cu CU_GCN_MP.cu CU_SpMM_GCN.cu cudaDataLoader.cu --compiler-options -fPIC`  
`nvcc --shared -o libCU_SpMM_GCN.so cudaDataLoader.o cuBlasUtil.o Data_Util.o CU_SpMM_GCN.o --compiler-options -fPIC -std=c++11`  
