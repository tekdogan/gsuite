#include<iostream>
#include<cmath>
#include"C_GCN_MP.h"
#include<cuda.h>
#include "scatter_cuda.h"
#include "linear.h"
#include "index_select.h"


namespace CU_MP {


float* GCNLayer(int* h_edgeIndex, float* h_featureVector, float *h_aggregationVar, float *h_nodeDegrees,
		int numOfNodes, int numOfFeatures, int numOfEdges, int outputSize) {

	// allocations for host variables
	//float *h_nodeDegrees = (float*)calloc(numOfNodes, sizeof(float));
	float *h_ones = (float*)calloc(numOfEdges, sizeof(float));
	
	// first part of edgeIndex indicating sources
	int *h_edgeSources = (int*)calloc(numOfEdges, sizeof(int));
//	memcpy(h_edgeSources, h_edgeIndex, numOfEdges*sizeof(int));
	for(int i = 0; i<numOfEdges; i++)
	{
		int d = h_edgeIndex[i];
		h_edgeSources[i] = d;
	}


	// ones to be used during node degree calculation
	//for(int i=0; i<numOfNodes; i++) {
	//	*(h_ones + i) = 1;
	//}
	memset(h_ones, 1, numOfEdges*sizeof(float));
	
	// compute the node degrees via scatter_add
	h_nodeDegrees = scatter_cuda(h_ones, h_edgeSources, 1, "sum", numOfEdges, numOfEdges, 1, numOfEdges, 1);

	// sqrt -0.5 of node degrees
	for(int i=0; i<numOfNodes; i++) {
		*(h_nodeDegrees + i) = 1/sqrt(*(h_nodeDegrees + i));
	}
	
	float *h_outputLinear = (float*)calloc(numOfNodes*outputSize, sizeof(float));

	// linear transform
	linear(h_featureVector, numOfNodes, numOfFeatures,
               h_outputLinear, numOfNodes, outputSize);

	// index select
	float *indexSelectOutput = (float*)calloc(numOfEdges*outputSize, sizeof(float));
	indexSelectOutput = index_select(h_outputLinear, numOfNodes, outputSize, 0, h_edgeSources, numOfEdges, indexSelectOutput);

	// aggregation via scatter
	int *h_edgeDest = (int*)calloc(numOfEdges, sizeof(int));
	memcpy(h_edgeDest, h_edgeIndex+numOfEdges, numOfEdges*sizeof(int));
	float *output = (float*)calloc(numOfNodes*outputSize, sizeof(float));
	output = scatter_cuda(indexSelectOutput, h_edgeDest, 1, "sum", numOfEdges, numOfEdges, 1, numOfEdges, outputSize);
	
	return output;
	
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
