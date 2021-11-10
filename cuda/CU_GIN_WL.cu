#include<iostream>
#include<cmath>
#include"CU_GIN_WL.h"
#include<omp.h>
#include<cuda.h>

namespace CU_WL {

__global__ void GINLayer(float* edgeIndex, float* featureTensor, float *aggregationVar, float epsilon, float* featureTensorOutput,
			int numOfNodes, int numOfDirectedEdges, int numOfFeatures, float* outputFeatureMatrix) {

    int i = threadIdx.x;
    if(i < numOfNodes) {
                for(int j=0; j<numOfDirectedEdges; j++) {
			if((*(edgeIndex + j)) == (float)i) { // if there is an edge incoming to node i
				for(int k=0; k<numOfFeatures; k++) {
					*(aggregationVar + i*numOfFeatures + k) += *(featureTensor + i*numOfFeatures + k);
				}
			}
                }

		for(int k=0; k<numOfFeatures; k++) {
			*(outputFeatureMatrix + i*numOfFeatures + k) = (1 + epsilon)*(*(outputFeatureMatrix + i*numOfFeatures + k)) + *(aggregationVar + i*numOfFeatures + k);
		}
    }
}

} // namespace end
