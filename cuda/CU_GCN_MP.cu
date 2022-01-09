#include<iostream>
#include<cmath>
#include"C_GCN_MP.h"
#include<omp.h>
#include<cuda.h>
#include "scatter_cuda.h"
#include "linear.h"
#include "index_select.h"


namespace CU_MP {


void GCNLayer(int* edgeIndex, float* featureVector, float *aggregationVar, float *nodeDegrees,
		int numOfNodes, int numOfFeatures, int numOfEdges, int outputSize) {

	

	// allocations for host variables
	float *h_edgeIndex = (float*)calloc(numOfEdges * 2, sizeof(float));
	float *h_featureVector = (float*)calloc(numOfNodes * numOfFeatures, sizeof(float));
	float *h_aggregationVar = (float*)calloc(numOfNodes * numOfFeatures, sizeof(float));
	float *h_nodeDegrees = (float*)calloc(numOfNodes, sizeof(float));
	float *h_ones = (float*)calloc(numOfNodes, sizeof(float));
	
	// first part of edgeIndex indicating sources
	int *h_edgeSources = (int*)calloc(numOfEdges, sizeof(int));
	memcpy(h_edgeSources, edgeIndex, numOfEdges*sizeof(int));

	// ones to be used during node degree calculation
	//for(int i=0; i<numOfNodes; i++) {
	//	*(h_ones + i) = 1;
	//}
	memset(h_ones, 1, numOfNodes*sizeof(float));
	
	// compute the node degrees via scatter_add
	h_nodeDegrees = scatter_cuda(h_nodeDegrees, h_edgeSources, 1, "sum", numOfEdges, numOfEdges, 1, numOfEdges, 1);

	// sqrt -0.5 of node degrees
	for(int i=0; i<numOfNodes; i++) {
		*(h_nodeDegrees + i) = 1/sqrt(*(h_nodeDegrees + i));
	}
	
	float *h_outputLinear = (float*)calloc(numOfNodes*outputSize, sizeof(float));

	// linear transform
	linear(featureVector, numOfNodes, numOfFeatures,
               h_outputLinear, numOfNodes, outputSize);

	int *edgeIndexSources = (int*)calloc(numOfEdges, sizeof(int));

	float *indexSelectOutput = (float*)calloc((numOfEdges)*outputSize, sizeof(float));
	indexSelectOutput = index_select(h_outputLinear, numOfNodes, outputSize, 0, edgeIndexSources, numOfEdges, indexSelectOutput);

	int *h_edgeDest = (int*)calloc(numOfEdges, sizeof(int));
	memcpy(h_edgeDest, edgeIndex+(numOfEdges), numOfEdges);
	float *output = scatter_cuda(indexSelectOutput, h_edgeDest, 1, "sum", numOfEdges, numOfEdges, 1, numOfEdges, outputSize);

	// aggregation scheme
	//auto out = scatter_cuda(h_featureVector, h_edgeIndex, 1, "sum", numOfNodes, numOfFeatures, numOfEdges);
	
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
