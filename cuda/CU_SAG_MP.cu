#include<iostream>
#include<cmath>
#include"CU_SAG_MP.h"
#include<omp.h>
#include<cuda.h>

#define DIRECTED_EDGES 4

#define NUM_NODES 3

#define FEATURE_LEN 3


namespace CU_MP {

__global__ void SAGLayer(float* edgeIndex, float* featureTensor, float *aggregationVar, float *nodeDegrees, float w1, float w2) {

	int i = threadIdx.x;
	if(i < NUM_NODES) {
                for(int j=0; j<DIRECTED_EDGES; j++) {
                        if((*(edgeIndex + j)) == (float)i) {// if there is an edge incoming to node i
                                // aggregate edgeIndex[1][j] features on node i
                                //std::cout << "from node " << edgeIndex[1][j] << " to node " << i << std::endl;
                                for(int k=0; k<FEATURE_LEN; k++) {
                                        aggregationVar[k] += *(featureTensor + i*FEATURE_LEN + k) * 1.0/sqrt(nodeDegrees[i]*nodeDegrees[(int)(*(edgeIndex + 1*DIRECTED_EDGES + j))]);
                                }
                        }
                }
                *(featureTensor + FEATURE_LEN*i) = aggregationVar[0];
                *(featureTensor + FEATURE_LEN*i + 1) = aggregationVar[1];
                aggregationVar[0] = 0.0;
                aggregationVar[1] = 0.0;
        }


}


} // namespace end
