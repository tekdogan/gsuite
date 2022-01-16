#include<iostream>
#include"CU_GIN_WL.h"
#include<cuda.h>
#include"index_select.h"
#include"linear.h"

#include "scatter_cuda.h"
#include "index_select.h"
#include "linear.h"

namespace CU_WL {


float* GINLayer(int* h_edgeIndex, float* h_featureVector, int numOfNodes, int numOfFeatures,
			int numOfEdges, int outputSize, float eps) {


	int* h_edgeIndexSrc;
	int* h_edgeIndexDst;
	h_edgeIndexSrc = (int*)calloc(numOfEdges, sizeof(int));
        h_edgeIndexDst = (int*)calloc(numOfEdges, sizeof(int));

	memcpy(h_edgeIndexSrc, h_edgeIndex, numOfEdges);
	memcpy(h_edgeIndexDst, h_edgeIndex + numOfEdges, numOfEdges);


        float* indexSelectOutput = index_select(h_featureVector, numOfNodes, numOfFeatures, 0, h_edgeIndexSrc, numOfEdges);


	float* aggrOutput = scatter_cuda(indexSelectOutput, h_edgeIndexDst, 1, "sum", numOfEdges, numOfEdges, numOfFeatures, numOfNodes, numOfFeatures);
	
	for(int i = 0; i<numOfNodes*numOfFeatures; i++) {
		aggrOutput[i] += (1 + eps) * h_featureVector[i];
	}


	float* output = linear(aggrOutput, numOfNodes, numOfFeatures, numOfNodes, outputSize);	

	
	free(h_edgeIndexSrc);
	free(h_edgeIndexDst);
	free(indexSelectOutput);
	free(aggrOutput);	

	return output;
}

} // namespace end
