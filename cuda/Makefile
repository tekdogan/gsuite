all:
	nvcc -g --cudart shared -lcublas_static -lculibos -ldl -lpthread -lcudart -std=c++11 -gencode arch=compute_61,code=compute_61 cuBlasUtil.cu Data_Util.cu CU_GCN_MP.cu CU_SpMM_GCN.cu cudaDataLoader.cu main.cpp -o cudaDataLoader.o
clean:
	rm cudaDataLoader.o
