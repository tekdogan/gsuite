namespace CU_WL {

__global__ void GINLayer(float* edgeIndex, float* featureTensor, float *aggregationVar, float epsilon, float* featureTensorOutput,
                        int numOfNodes, int numOfDirectedEdges, int numOfFeatures, float* outputFeatureMatrix);

}
