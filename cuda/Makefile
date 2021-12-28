all:
	nvcc -g --cudart shared -lcublas_static -lculibos -ldl -lpthread -lcudart -lcudadevrt -std=c++11 -gencode arch=compute_61,code=compute_61 cuBlasUtil.cu Data_Util.cu  CU_GCN_MP.cu CU_SpMM_GCN.cu CU_SpMM_GIN.cu CU_SAG_WL.cu CU_GIN_WL.cu cudaDataLoader.cu main.cpp -o cudaDataLoader.o
clean:
	rm cudaDataLoader.o
