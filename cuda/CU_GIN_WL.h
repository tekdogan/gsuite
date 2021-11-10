namespace CU_WL {

__global__ void GINLayer(float* edgeIndex, float* featureTensor, float *aggregationVar, float epsilon,
                        int numOfNodes, int numOfDirectedEdges, int numOfFeatures, float* outputFeatureMatrix);

}
