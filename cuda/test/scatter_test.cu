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
        float h_src[48] = 
	{
		    0,0,0,0,
		    0,0,0,0,
		    0,0,0,0,
                    1,1,1,1,
		    1,1,1,1,
                    2,2,2,2,
                    2,2,2,2,
		    2,2,2,2,
                    3,3,3,3,
		    3,3,3,3,
		    4,4,4,4,
		    4,4,4,4
	};
	
	int featureLen = 4;
	int nodeCount = 5;
	int edgeCount = 12;	

	float h_edgeSource[12] = {0,0,0,1,1,2,2,2,3,3,4,4};
	float h_edgeDest[12] = {1,2,4,0,3,0,3,4,1,2,0,2};

        int srcRows = 5;
        int srcCols = 4;

        int dstSize = nodeCount * featureLen;


	float* d_src;
	float* d_index;


	cudaMalloc((void**) &d_src, (featureLen * edgeCount)*sizeof(float));
	cudaMemcpy(d_src, h_src, (featureLen * edgeCount)*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_index, (edgeCount)*sizeof(float));
        cudaMemcpy(d_index, h_edgeDest, (edgeCount)*sizeof(float), cudaMemcpyHostToDevice);

	float* a = scatter_cuda(d_src, d_index, 1, "sum", nodeCount, featureLen, edgeCount);

	sleep(10);

	
        float *h_out = (float*)calloc(nodeCount*featureLen, sizeof(float));

	cudaMemcpy(h_out, a, (nodeCount*featureLen*sizeof(float)), cudaMemcpyDeviceToHost);

	printDenseMatrix(a, nodeCount, featureLen);

        return 0;
}


