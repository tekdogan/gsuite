// This function is designed for a temporary use of
// direct usage of CUDA data loader utility from
// a cpp based main file.

//#include"DataLoader.h"
#include "../scatter_cuda.h"
#include "../Data_Util.h"
#include <iostream>

#include <unistd.h>

/*

edge_index 
source 0,0,0,1,1,2,2,2,3,3,4,4
dest   1,2,4,0,3,0,3,4,1,2,0,2
*/

int main(int argc, char *argv[]) {

        //if(argc == 1) {
        //      std::cout << "Please pass a parameter to executable. (e.g. ./cudaDataLoader.o 2)\n";
        //}

        //LoadData(atoi(argv[1]));

	/*
		3x4 node to feature vector
	*/

        float h_src[12] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

	int nodeCount = 5;
	int edgeCount = 12;	

	float *h_out = (float*)calloc(nodeCount,sizeof(float));

	int h_edgeSource[12] = {0,0,0,1,1,2,2,2,3,3,4,4};
	int h_edgeDest[12] = {1,2,4,0,3,0,3,4,1,2,0,2};


	float* out = scatter_cuda(h_src, h_edgeSource, 1, "sum", edgeCount, 1, edgeCount, 1, nodeCount);

	sleep(3);

	printDenseMatrix(out, 1, nodeCount);

        return 0;
}


