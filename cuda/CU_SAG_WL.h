namespace CU_WL {

__global__ void SAGLayer(float* edgeIndex, float* featureTensor, float w1, float w2, int numOfNodes, int numOfEdges, int numOfFeatures, float* tempFeatureValues, float* outputFeatureMatrix);

}
