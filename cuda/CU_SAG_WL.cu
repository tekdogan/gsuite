#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cuda.h>
#include <cublas_v2.h>
#include "CU_SAG_WL.h"
#include "cuBlasUtil.h"
#include "Data_Util.h"

namespace CU_WL {

__global__ void SAGLayer(float* edgeIndex, float* featureTensor, float w1, float w2, int numOfNodes, int numOfEdges, int numOfFeatures, float* outputFeatureMatrix) {

	printf("thread\n");
	int i = threadIdx.x;
	printf("thread %d\n",i);
	if(i < numOfNodes) {
	    
	        // temporary feature values variable used during
                // the calculation of mean values of incoming edges
                float* tempFeatureValues;
		cudaMalloc(&tempFeatureValues, numOfFeatures * sizeof(float));
                        
                // number of incoming edges to i
                int tempIncomingEdges = 0;
	    
	            // scan through edge indices
                for(int j=0; j<numOfEdges; j++) {
                
                        if((*(edgeIndex + j)) == (float)i) { // if there is an edge incoming to node i
                                
                                // add xj values to sum
                                for(int k=0; k<numOfFeatures; k++) {
                                        *(tempFeatureValues + k) += *(featureTensor + ((int)*(edgeIndex + numOfEdges + j))*numOfFeatures + k);
                                }
                                
                                // increment number of incoming edges to node i
                                tempIncomingEdges++;
                        }
                }
                
                // calculate new values of node features of i
                for(int k=0; k<numOfFeatures; k++) {
			*(outputFeatureMatrix + i*numOfFeatures + k) = (w1 * *(outputFeatureMatrix + i*numOfFeatures + k)) + (w2 * (tempFeatureValues[k]/tempIncomingEdges));
			printf("calculated value of node %d feature %d is %f\n",i,k,(w1 * *(outputFeatureMatrix + i*numOfFeatures + k)) + (w2 * (tempFeatureValues[k]/tempIncomingEdges)) );
		}
		cudaFree(tempFeatureValues);
        }


}

} // namespace end

