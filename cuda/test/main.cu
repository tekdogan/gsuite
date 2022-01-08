// This function is designed for a temporary use of
// direct usage of CUDA data loader utility from
// a cpp based main file.

//#include"DataLoader.h"
#include "index_select.h"
#include "Data_Util.h"
#include <iostream>

int main(int argc, char *argv[]) {

	//if(argc == 1) {
	//	std::cout << "Please pass a parameter to executable. (e.g. ./cudaDataLoader.o 2)\n";
	//}

	//LoadData(atoi(argv[1]));

	float h_src[9] = {1,2,3,
		    4,5,6,
		    7,8,9};

	int srcRows = 3;
	int srcCols = 3;

	int dim = 0;

	int h_indices[2] = {0,2};

	int indSize = 2;

	int dstSize = 6;

	float *h_out = (float*)calloc(dstSize, sizeof(float));

	float *d_src, *d_out;
	int *d_indices;

	cudaMalloc((void**) &d_src, srcRows*srcCols*sizeof(float));
	cudaMemcpy(d_src, h_src, srcRows*srcCols*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_out, dstSize*sizeof(float));
        cudaMemcpy(d_out, h_out, dstSize*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_indices, indSize*sizeof(int));
	cudaMemcpy(d_indices, h_indices, indSize*sizeof(int), cudaMemcpyHostToDevice);

	index_select(h_src, srcRows, srcCols,
                    dim, h_indices, indSize, h_out);

	//cudaMemcpy(h_out, d_out, dstSize*sizeof(float), cudaMemcpyDeviceToHost);

	printDenseMatrix(h_out, dstSize/srcCols, srcCols);

	return 0;
}
