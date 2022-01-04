#include<iostream>
#include<cmath>
#include"C_GCN_MP.h"
#include<omp.h>
#include<cuda.h>
#include "scatter_cuda.h"


namespace CU_MP {


void GCNLayer(float* d_edgeIndex, float* d_featureVector, float *d_aggregationVar, float *d_nodeDegrees,
		int numOfNodes, int numOfFeatures, int numOfEdges) {

	
	// variable declarations for host variables
	float *h_edgeIndex, *h_featureVector, *h_aggregationVar, *h_nodeDegrees;

	// allocations for host variables
	*h_edgeIndex = calloc(numOfEdges * 2, sizeof(float));
	*h_featureVector = calloc(numOfNodes * numOfFeatures, sizeof(float));
	*h_aggregationVar = calloc(numOfNodes * numOfFeatures, sizeof(float));
	*h_nodeDegrees = calloc(numOfNodes, sizeof(float));
	
	//cudaMalloc( (void**) &d_featureVector, numOfNodes*numOfFeatures * sizeof(float));
	//cudaMemcpy(d_featureVector, h_featureVector, numOfNodes*numOfFeatures * sizeof(float), cudaMemcpyHostToDevice);

	
	// compute the node degrees via scatter_add
	auto res = scatter_cuda(h_nodeDegrees, h_edgeIndex, 1, "sum", numOfNodes, numOfFeatures, numOfEdges);

	// migrate device degrees output to host
	cudaMemcpy(d_nodeDegrees, h_nodeDegrees, numOfNodes * sizeof(float), cudaMemcpyHostToDevice);
	
	// sqrt -0.5 of node degrees
	for(int i=0; i<numOfNodes; i++) {
		*(h_nodeDegrees + i) = 1/sqrt(*(h_nodeDegrees + i));
	}
	
	// migrate host degrees back to device
	cudaMemcpy(h_nodeDegrees, d_nodeDegrees, numOfNodes * sizeof(float), cudaMemcpyDeviceToHost);
	
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
