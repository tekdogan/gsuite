namespace CU_WL {

__global__ void SAGLayer(float* edgeIndex, float* featureTensor, float w1, float w2, int numOfNodes, int numOfDirectedEdges,
                         int numOfFeatures, float* tempFeatureValues, int* tempIncomingEdges, float* outputFeatureMatrix);

__global__ void SAGLayer2(float* edgeIndex, float* featureTensor, float w1, float w2, int numOfNodes, int numOfEdges, int numOfFeatures, float* tempFeatureValues, float* outputFeatureMatrix);

}
