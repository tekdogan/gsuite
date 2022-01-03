#include<iostream>
#include<cmath>
#include"C_GCN_MP.h"
#include<omp.h>
#include<cuda.h>
#include "scatter_cuda.h"

//#define DIRECTED_EDGES 4

//#define NUM_NODES 3

//#define FEATURE_LEN 3


namespace CU_MP {


void GCNLayer(float* edgeIndex, float* featureTensor, float *aggregationVar, float *nodeDegrees,
		int numOfNodes, int numOfFeatures, int numOfEdges) {

	//int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// compute the node degrees
	auto res = scatter_cuda(nodeDegrees, edgeIndex, 1, "add", numOfNodes, featureSize, edgeIndexSize);
	
	// sqrt -0.5 of node degrees
	for(int i=0; i<numOfNodes; i++) {
		*(numOfNodes + i) = 1/sqrt(numOfNodes);
	}
	
	// aggregation scheme
	auto out = scatter_cuda(featureTensor, edgeIndex, 1, "add", numOfNodes, featureSize, edgeIndexSize);
	
	/*if (thread_idx < numOfNodes*numOfFeatures) {
		
		const int64_t id_exEdges = (thread_idx % numOfEdges);
		
		const int64_t id_exNodes = (thread_idx / numOfFeatures);
		
		const int64_t id_exFeatures = (thread_idx / numOfNodes);
		
		// if an incoming edge to respected node
		if( *(edgeIndex + numOfEdges + id_exEdges) == id_exNodes ) {
			// then apply aggregation scheme of GCN
			// to corresponding node's feature
			*(aggregationVar + (int)numOfFeatures*( (int)*(edgeIndex + numOfEdges + id_exEdges) )
			  + id_exFeatures) += *(featureTensor + thread_idx) *
				1.0/sqrt(nodeDegrees[id_exNodes]*
					 nodeDegrees[( (int)*(edgeIndex + numOfEdges + id_exEdges) )]);
		}
		
	}*/
	
}


__global__ void GCNLayerNew(float* edgeIndex, float* featureTensor, float *aggregationVar, float *nodeDegrees, int numOfNodes, int numOfFeatures, int numOfEdges) {

	int i = threadIdx.x;
	int j = blockIdx. x;
	int k = j;
	//if(i < numOfNodes) {
                //for(int j=0; j<numOfEdges; j++) {
                        if((*(edgeIndex + j)) == (float)i) {// if there is an edge incoming to node i
                                // aggregate edgeIndex[1][j] features on node i
                                //std::cout << "from node " << edgeIndex[1][j] << " to node " << i << std::endl;
                                //for(int k=0; k<numOfFeatures; k++) {
                                        *(aggregationVar + i*numOfFeatures + k) += *(featureTensor + i*numOfFeatures + k) * 1.0/sqrt(nodeDegrees[i]*nodeDegrees[(int)(*(edgeIndex + 1*numOfEdges + j))]);
                                //}
                        }
                //}
                //*(featureTensor + numOfFeatures*i) = *(aggregationVar + i*numOfFeatures);
                //*(featureTensor + numOfFeatures*i + 1) = *(aggregationVar + i*numOfFeatures + 1);
                //*(aggregationVar + i*numOfFeatures) = 0.0;
                //*(aggregationVar + i*numOfFeatures + 1) = 0.0;
        //}


}


} // namespace end
