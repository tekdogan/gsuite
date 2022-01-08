// This function is designed for a temporary use of
// direct usage of CUDA data loader utility from
// a cpp based main file.

//#include"DataLoader.h"
#include "linear.h"
#include "Data_Util.h"
#include <iostream>

int main(int argc, char *argv[]) {

	//if(argc == 1) {
	//	std::cout << "Please pass a parameter to executable. (e.g. ./cudaDataLoader.o 2)\n";
	//}

	//LoadData(atoi(argv[1]));

	float h_src[6] = {1,2,
		          3,4,
		          5,6};

	int srcRows = 3;
	int srcCols = 2;

	int outRows = srcRows;
	int outCols = 1;

	float *h_out = (float*)calloc(outRows*outCols, sizeof(float));

	float *d_src, *d_out;

	//cudaMalloc((void**) &d_src, srcRows*srcCols*sizeof(float));
	//cudaMemcpy(d_src, h_src, srcRows*srcCols*sizeof(float), cudaMemcpyHostToDevice);

	//cudaMalloc((void**) &d_out, dstSize*sizeof(float));
        //cudaMemcpy(d_out, h_out, dstSize*sizeof(float), cudaMemcpyHostToDevice);

	//cudaMalloc((void**) &d_indices, indSize*sizeof(int));
	//cudaMemcpy(d_indices, h_indices, indSize*sizeof(int), cudaMemcpyHostToDevice);

	linear(h_src, srcRows, srcCols,
              h_out, outRows, outCols);

	//cudaMemcpy(h_out, d_out, dstSize*sizeof(float), cudaMemcpyDeviceToHost);

	//printDenseMatrix(h_out, dstSize/srcCols, srcCols);

	return 0;
}
