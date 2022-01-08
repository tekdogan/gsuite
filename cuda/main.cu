// This function is designed for a temporary use of
// direct usage of CUDA data loader utility from
// a cpp based main file.

//#include"DataLoader.h"
#include "index_select.h"

int main(int argc, char *argv[]) {

	//if(argc == 1) {
	//	std::cout << "Please pass a parameter to executable. (e.g. ./cudaDataLoader.o 2)\n";
	//}

	//LoadData(atoi(argv[1]));

	indexSelectLargeIndex(float *src, int srcRows, int srcCols,
                    int dim, int* indices, int indSize, int dstSize,
                    float *out);

	return 0;
}
