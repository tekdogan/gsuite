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

__global__ void SAGLayer(float* edgeIndex, float* featureTensor, float w1, float w2, int numOfNodes, int numOfEdges,
			 int numOfFeatures, float* tempFeatureValues, int* tempIncomingEdges, float* outputFeatureMatrix) {

	int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (thread_idx < numOfNodes*numOfFeatures*numOfDirectedEdges) {
		
		printf("thread_idx is: %d\n", thread_idx);
		
		printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d\n", blockIdx.x, blockDim.x, threadIdx.x);
		
		const int64_t id_exEdges = (thread_idx / numOfNodes * numOfFeatures);
		
		const int64_t id_exNodes = (thread_idx / numOfDirectedEdges * numOfFeatures);
		
		const int64_t id_exFeatures = (thread_idx / numOfNodes * numOfDirectedEdges);
		
		// if an incoming edge to respected node
		if( *(edgeIndex + numOfDirectedEdges + id_exEdges) == id_exNodes ) {
			// apply aggregation of the neighbour node's to temporary
			// feature vector analogus to SAG formula
			*(tempFeatureMatrix + numOfFeatures*( *(edgeIndex + numOfDirectedEdges + id_exEdges) )
				+ id_exFeatures) += *(src + thread_idx);
			
			// increment number of incoming edges to corresponding node
			tempIncomingEdges[id_exNodes]++;
		}
		
		// sync threads before output update
		// __syncthreads();
		
		// update output matrix
		*(outputFeatureMatrix + numOfFeatures*id_exNodes + id_exFeatures) =
			(w1 * *(featureTensor + numOfFeatures*id_exNodes + id_exFeatures)) +
			(w2 * (*(tempFeatureValues + numOfNodes*id_exNodes + id_exFeatures)/tempIncomingEdges[id_exNodes]));
		
	}
	
	// below operations are going to be removed after the
	// kernel update with new computation model
	//if(i < numOfNodes) {
	    
	        // temporary feature values variable used during
                // the calculation of mean values of incoming edges
                //float* tempFeatureValues;
		//cudaMalloc(&tempFeatureValues, numOfFeatures * sizeof(float));

		// escape variable to prevent memory dependency
		//bool esc_var = false;
                        
                // number of incoming edges to i
                int tempIncomingEdges = 0;
	        //printf("DEBUG: CU_WL::SAGLayer first part starting...\n");
	        // scan through edge indices
                //for(int j=0; j<numOfEdges; j++) {

                        if((*(edgeIndex + j)) == (float)i) { // if there is an edge incoming to node i
                                
				//printf("DEBUG: thread[%d] CU_WL::SAGLayer inside incoming edge\n", i);

                                // add xj values to sum
                                //for(int k=0; k<numOfFeatures; k++) {
                                        *(tempFeatureValues + i*numOfNodes + k) += *(featureTensor + ((int)*(edgeIndex + numOfEdges + j))*numOfFeatures + k);
                                //}
                                
                                // increment number of incoming edges to node i
                                tempIncomingEdges++;

				// set escape variable
				//esc_var = true;
                        }
			//else if(esc_var) {
			//	// escape if thread's turn is over
			//	//printf("thread %d escaping at %d\n", i, j);
			//	break;
			//}
			//else {
			//	//printf("DEBUG: else\n");
			//}
                //}
                //printf("DEBUG: CU_WL::SAGLayer THREAD:%d first part successful!\n", i);

		//printf("DEBUG: numOfFeatures is %d\n", numOfFeatures);
                // calculate new values of node features of i
                //for(int k=0; k<numOfFeatures; k++) {
			*(outputFeatureMatrix + i*numOfFeatures + k) = (w1 * *(featureTensor + i*numOfFeatures + k)) + (w2 * (*(tempFeatureValues + i*numOfNodes + k)/tempIncomingEdges));
			//printf("DEBUG: CU_WL::SAGLayer inside the aggregation part.\n");
			//printf("DEBUG: CU_WL::SAGLayer THREAD:%d calculated value of node %d feature %d is %f\n",i,i,k,(w1 * *(outputFeatureMatrix + i*numOfFeatures + k)) + (w2 * (*(tempFeatureValues + i*numOfNodes + k)/tempIncomingEdges)) );
		//}
		//cudaFree(tempFeatureValues);
		
        //}


}


__global__ void SAGLayer2(float* edgeIndex, float* featureTensor, float w1, float w2, int numOfNodes, int numOfEdges, int numOfFeatures, float* tempFeatureValues, float* outputFeatureMatrix) {

        int i = threadIdx.x + blockIdx.x * blockDim.x;
        //printf("thread %d\n",i);
        if(i < numOfNodes) {

                // temporary feature values variable used during
                // the calculation of mean values of incoming edges
                //float* tempFeatureValues;
                //cudaMalloc(&tempFeatureValues, numOfFeatures * sizeof(float));

                // number of incoming edges to i
                int tempIncomingEdges = 0;
                //printf("DEBUG: CU_WL::SAGLayer first part starting...\n");
                // scan through edge indices
                for(int j=0; j<numOfEdges; j++) {

                        if((*(edgeIndex + j*2)) == (float)i) { // if there is an edge incoming to node i

                                // add xj values to sum
                                for(int k=0; k<numOfFeatures; k++) {
                                        *(tempFeatureValues + i*numOfNodes + k) += *(featureTensor + ((int)*(edgeIndex + 2*j))*numOfFeatures + k);
                                }

                                // increment number of incoming edges to node i
                                tempIncomingEdges++;
                        }
                }
                //printf("DEBUG: CU_WL::SAGLayer THREAD:%d first part successful!\n", i);

                //printf("DEBUG: numOfFeatures is %d\n", numOfFeatures);
                // calculate new values of node features of i
                for(int k=0; k<numOfFeatures; k++) {
                        *(outputFeatureMatrix + i*numOfFeatures + k) = (w1 * *(featureTensor + i*numOfFeatures + k)) + (w2 * (*(tempFeatureValues + i*numOfNodes + k)/tempIncomingEdges));
                        //printf("DEBUG: CU_WL::SAGLayer inside the aggregation part.\n");
                        //printf("DEBUG: CU_WL::SAGLayer THREAD:%d calculated value of node %d feature %d is %f\n",i,i,k,(w1 * *(outputFeatureMatrix + i*numOfFeatures + k)) + (w2 * (*(tempFeatureValues + i*numOfNodes + k)/tempIncomingEd$
                }
                //cudaFree(tempFeatureValues);

        }


}

} // namespace end

