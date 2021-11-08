#include<iostream>
#include<cmath>
#include"C_GCN_MP.h"
#include<omp.h>
#include<cuda.h>

//#define DIRECTED_EDGES 4

//#define NUM_NODES 3

//#define FEATURE_LEN 3


namespace CU_MP {

void GCNLayer(float* edgeIndex, float* featureTensor, float* aggregationVar, float* nodeDegrees) {
/*
        for(int i=0; i<NUM_NODES; i++) {
                for(int j=0; j<DIRECTED_EDGES; j++) {
                        if(edgeIndex[0][j] == i) {// if there is an edge incoming to node i
                                // aggregate edgeIndex[1][j] features on node i
                                //std::cout << "from node " << edgeIndex[1][j] << " to node " << i << std::endl;
                                for(int k=0; k<FEATURE_LEN; k++) {
                                        aggregationVar[k] += featureTensor[i][k] * 1.0/sqrt(nodeDegrees[i]*nodeDegrees[(int)edgeIndex[1][j]]);
                                }
                        }
                }
                featureTensor[i][0] = aggregationVar[0];
                featureTensor[i][1] = aggregationVar[1];
                aggregationVar[0] = 0.0;
                aggregationVar[1] = 0.0;
        }
*/
}

__global__ void GCNLayerNew(float* edgeIndex, float* featureTensor, float *aggregationVar, float *nodeDegrees, int numOfNodes, int numOfFeatures, int numOfEdges) {

	int i = threadIdx.x;
	if(i < numOfNodes) {
                for(int j=0; j<numOfEdges; j++) {
                        if((*(edgeIndex + j)) == (float)i) {// if there is an edge incoming to node i
                                // aggregate edgeIndex[1][j] features on node i
                                //std::cout << "from node " << edgeIndex[1][j] << " to node " << i << std::endl;
                                for(int k=0; k<numOfFeatures; k++) {
                                        *(aggregationVar + i*numOfFeatures + k) += *(featureTensor + i*numOfFeatures + k) * 1.0/sqrt(nodeDegrees[i]*nodeDegrees[(int)(*(edgeIndex + 1*numOfEdges + j))]);
                                }
                        }
                }
                *(featureTensor + numOfFeatures*i) = *(aggregationVar + i*numOfFeatures);
                *(featureTensor + numOfFeatures*i + 1) = *(aggregationVar + i*numOfFeatures + 1);
                *(aggregationVar + i*numOfFeatures) = 0.0;
                *(aggregationVar + i*numOfFeatures + 1) = 0.0;
        }


}


} // namespace end
