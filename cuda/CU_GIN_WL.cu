#include<iostream>
#include<cmath>
#include"C_GCN_MP.h"
#include<omp.h>
#include<cuda.h>


__global__ void GINLayer(float* edgeIndex, float* featureTensor, float *aggregationVar, float *nodeDegrees, float epsilon, float* featureTensorOutput,
			int numOfNodes, int numOfDirectedEdges, int numOfFeatures) {

    int i = threadIdx.x;
    if(i < numOfNodes) {
                for(int j=0; j<DIRECTED_EDGES; j++) {
                        if((*(edgeIndex + j)) == (float)i) {// if there is an edge incoming to node i
                                // aggregate edgeIndex[1][j] features on node i
                                //std::cout << "from node " << edgeIndex[1][j] << " to node " << i << std::endl;
                                for(int k=0; k<numOfFeatures; k++) {
                                        aggregationVar[k] += *(featureTensor + numOfFeatures*((int)(*(edgeIndex + 1*numOfDirectedEdges + j))) + j);
                                }
                        }
                }
                *(featureTensorOutput + numOfFeatures*i) = aggregationVar[0] + (*(featureTensor + numOfFeatures*i))*(1+epsilon);
                *(featureTensorOutput + numOfFeatures*i + 1) = aggregationVar[1] + (*(featureTensor + numOfFeatures*i + 1))*(1+epsilon);
                aggregationVar[0] = 0.0;
                aggregationVar[1] = 0.0;
        }
}
