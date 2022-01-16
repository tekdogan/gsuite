#include<iostream>
#include<cmath>
#include"CU_SAG_MP.h"
#include<omp.h>
#include<cuda.h>

#include "index_select.h"
#include "scatter_cuda.h"
#include "linear.h"

namespace CU_MP {

float* SAGELayer(int* h_edgeIndex, float* h_featureVector, int numOfNodes, int numOfFeatures,
                        int numOfEdges, int outputSize) {


        int* h_edgeIndexSrc;
        int* h_edgeIndexDst;
        h_edgeIndexSrc = (int*)calloc(numOfEdges, sizeof(int));
        h_edgeIndexDst = (int*)calloc(numOfEdges, sizeof(int));

        memcpy(h_edgeIndexSrc, h_edgeIndex, numOfEdges);
        memcpy(h_edgeIndexDst, h_edgeIndex + numOfEdges, numOfEdges);


        float* indexSelectOutput = index_select(h_featureVector, numOfNodes, numOfFeatures, 0, h_edgeIndexSrc, numOfEdges);

        float* aggrOutput = scatter_cuda(indexSelectOutput, h_edgeIndexDst, 1, "sum", numOfEdges, numOfEdges, numOfFeatures, numOfNodes, numOfFeatures);

        float* aggrTransformed = linear(aggrOutput, numOfNodes, numOfFeatures, numOfNodes, outputSize);

	float* inputTransformed = linear(h_featureVector, numOfNodes, numOfFeatures, numOfNodes, outputSize);

	for(int i = 0; i < numOfNodes*outputSize; i++) {
		aggrTransformed[i] += inputTransformed[i];
	}

        free(h_edgeIndexSrc);
        free(h_edgeIndexDst);
        free(indexSelectOutput);
	free(inputTransformed);

        return aggrTransformed;	
}


} // namespace end
