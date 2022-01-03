# GNN BM Suite (draft)

#### :information_source: This repository incorporates the material about in-progress benchmark suite for Graph Neural Networks.

<a href="https://github.com/tekdogan/gcn/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/tekdogan/gcn?style=plastic" /></a>


<a href="">
<img src="https://img.shields.io/github/stars/tekdogan/gcn.svg"/></a>

<a href="">
<img src="https://img.shields.io/github/commit-activity/w/tekdogan/gcn?style=plastic"/></a>

<a href="">
<img src="https://img.shields.io/github/followers/tekdogan?style=social"/></a>

<a href="">
        <img src="https://img.shields.io/badge/NVIDIA-V100-76B900?style=for-the-badge&logo=nvidia&logoColor=white" /></a>

### UI Parameters
`--config`: (_mandatory_) local configuration file which includes default parameters  
`--gpu-id` : (_optional_) device id  
`--model` : (_optional_) GNN model  
`--dataset` : (_optional_) dataset to be processed  

### Before Compiling
`export PATH=/usr/local/cuda-8.0/bin:$PATH`  
`export CPATH=/usr/local/cuda-8.0/include:$CPATH`  
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/usr/local/cuda-8.0/lib64/`  

### Profile via NVPROF
`nvprof -f --analysis-metrics -o cudaDataLoader.nvprof ./cudaDataLoader.o --benchmark`  

### lib compilation
`nvcc -lcublas -std=c++11 -c -arch=compute_61 cuBlasUtil.cu Data_Util.cu CU_GCN_MP.cu CU_SpMM_GCN.cu cudaDataLoader.cu --compiler-options -fPIC`  
`nvcc --shared -o libCU_SpMM_GCN.so cudaDataLoader.o cuBlasUtil.o Data_Util.o CU_SpMM_GCN.o --compiler-options -fPIC -std=c++11`  
