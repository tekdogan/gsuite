#include<iostream>
#include<cmath>
#include"C_GCN_MP.h"
#include<cuda.h>
#include "scatter_cuda.h"
#include "linear.h"
#include "index_select.h"


namespace CU_MP {


float* GCNLayer(int* h_edgeIndex, float* h_featureVector,
                int numOfNodes, int numOfFeatures,
                int numOfEdges, int outputSize) {

	// allocations for host variables
	float *h_ones = (float*)calloc(numOfEdges, sizeof(float));
	
	// first part of edgeIndex indicating sources
	int *h_edgeSources = (int*)calloc(numOfEdges, sizeof(int));
	for(int i = 0; i<numOfEdges; i++)
	{
		int d = h_edgeIndex[i];
		h_edgeSources[i] = d;
	}


	memset(h_ones, 1, numOfEdges*sizeof(float));
	
	// compute the node degrees via scatter_add
	float* h_nodeDegrees = scatter_cuda(h_ones, h_edgeSources, 1, "sum", numOfEdges, numOfEdges, 1, numOfEdges, 1);

	// sqrt -0.5 of node degrees
	for(int i=0; i<numOfNodes; i++) {
		*(h_nodeDegrees + i) = 1/sqrt(*(h_nodeDegrees + i));
	}
	
	
	// linear transform
	float* h_outputLinear = linear(h_featureVector, numOfNodes, numOfFeatures, numOfNodes, outputSize);

	// index select
	float* indexSelectOutput = index_select(h_outputLinear, numOfNodes, outputSize, 0, h_edgeSources, numOfEdges);



	// aggregation via scatter
	int *h_edgeDest = (int*)calloc(numOfEdges, sizeof(int));
	memcpy(h_edgeDest, h_edgeIndex+numOfEdges, numOfEdges*sizeof(int));
	float *output = (float*)calloc(numOfNodes*outputSize, sizeof(float));
	output = scatter_cuda(indexSelectOutput, h_edgeDest, 1, "sum", numOfEdges, numOfEdges, outputSize, numOfNodes, outputSize);
	

	free(h_ones);
	free(h_edgeSources);
	free(h_nodeDegrees);
	free(h_outputLinear);
	free(indexSelectOutput);
	free(h_edgeDest);
	return output;
}


} // namespace end
