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

	float *src = (float*)calloc(100*100, sizeof(float));
	float *out = (float*)calloc(100*100, sizeof(float));
	
	int srcRows = 3;
	int srcCols = 2;

	int outRows = srcRows;
	int outCols = 1;

	float *h_out = (float*)calloc(outRows*outCols, sizeof(float));

	//linear(h_src, srcRows, srcCols,
        //      h_out, outRows, outCols);
	
	linear(src, 100, 100,
              out, 100, 100);

	printDenseMatrix(h_src, outRows, outCols);

	return 0;
}
