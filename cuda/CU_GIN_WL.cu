#include<iostream>
#include<cmath>
#include"CU_GIN_WL.h"
#include<omp.h>
#include<cuda.h>

namespace CU_WL {

__global__ void GINLayer(float* edgeIndex, float* featureTensor, float *aggregationVar, float epsilon,
			int numOfNodes, int numOfDirectedEdges, int numOfFeatures, float* outputFeatureMatrix) {

    int i = threadIdx.x;
    int j= blockIdx.x;
    int k = 150;

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    //printf("thread_idx is: %d\n", thread_idx);

    if (thread_idx < numOfNodes*numOfFeatures*numOfDirectedEdges) {
    
    // get indices of the thread
    
    printf("thread_idx is: %d\n", thread_idx);

    printf("blockIdx.x: %d, blockDim.x: %d, threadIdx.x: %d\n", blockIdx.x, blockDim.x, threadIdx.x);

    const int64_t id_exEdges = (thread_idx / numOfNodes * numOfFeatures);

    const int64_t id_exNodes = (thread_idx / numOfDirectedEdges * numOfFeatures);
	    
    const int64_t id_exFeatures = (thread_idx / numOfNodes * numOfDirectedEdges);

    // if an incoming edge to respected node
    if( *(edgeIndex + numOfDirectedEdges + id_exEdges) == id_exNodes )
	*(featureTensor + numOfFeatures*( *(edgeIndex + numOfDirectedEdges + id_exEdges) )
	    + id_exFeatures) = *(src + thread_idx);
    }



    // the below part is on hold due to kernel update
    //if(i < numOfNodes) {
                //for(int j=0; j<numOfDirectedEdges; j++) {
			//if((*(edgeIndex + j)) == (float)i) { // if there is an edge incoming to node i
				//for(int k=0; k<numOfFeatures; k++) {
					//*(aggregationVar + i*numOfFeatures + k) += *(featureTensor + i*numOfFeatures + k);
				//}
			//}
                //}

		//for(int k=0; k<numOfFeatures; k++) {
			//*(outputFeatureMatrix + i*numOfFeatures + k) = (1 + epsilon)*(*(outputFeatureMatrix + i*numOfFeatures + k)) + *(aggregationVar + i*numOfFeatures + k);
		//}
    //}
}

} // namespace end
